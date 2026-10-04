// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parser logs must never carry request or model text.
//!
//! Deployments ship these logs to shared storage, and zero-data-retention
//! customers require that no prompt or completion text lands there. These tests
//! drive every registered tool-call family, every registered unified family
//! (native and guided, every starting state and invalid-payload policy) and the
//! vendored batch parsers with EOF recovery on and off, over well-formed,
//! truncated and malformed inputs in which every content position (prose,
//! reasoning, tool names, argument keys and values, schema type names, the named
//! tool choice) carries a sentinel. One test captures every `tracing` event at
//! TRACE; the other re-runs the corpus in a child process with
//! `DYNAMO_PARSERS_DEBUG=1` and captures its stderr. Both fail if the sentinel,
//! or a sentinel token id, appears in the output.

use std::io::Write;
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;

use super::traits::Tool;
use super::v1core::{
    Glm47ParserConfig, KimiK2ParserConfig, MiniMaxM3ParserConfig, ToolDefinition, XmlParserConfig,
    try_tool_call_parse_glm47, try_tool_call_parse_kimi_k2, try_tool_call_parse_minimax_m3,
    try_tool_call_parse_xml,
};
use super::{REGISTERED_FAMILIES, create_tool_parser_for_family};
use crate::unified::{
    InvalidGuidedPayloadPolicy, REGISTERED_UNIFIED_FAMILIES, UnifiedParserInit,
    UnifiedParserOutput, UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
    canonical_unified_family, create_unified_parser_for_family,
};

/// Every content position carries this marker. The check is case-insensitive.
const SENTINEL: &str = "ZDRSENTINEL";
/// A Harmony vocabulary id fed as model output.
const TOKEN_SENTINEL: u32 = 187_345;
/// An id outside the Harmony vocabulary, so decoding it fails.
const BAD_TOKEN_SENTINEL: u32 = 4_000_000_007;

const NAME: &str = "ZDRSENTINEL_tool";
const ROGUE: &str = "ZDRSENTINEL_rogue";
const OTHER: &str = "ZDRSENTINEL_other";
const BEFORE: &str = "ZDRSENTINEL prose before. ";
const AFTER: &str = " ZDRSENTINEL prose after.";
const THINKING: &str = "ZDRSENTINEL thinking";

/// Arguments: one per schema type, each value deliberately not of that type,
/// so every coercion fallback runs.
const ARGS: &[(&str, &str)] = &[
    ("ZDRSENTINEL_str", "ZDRSENTINEL string value"),
    ("ZDRSENTINEL_int", "ZDRSENTINEL not an integer"),
    ("ZDRSENTINEL_num", "ZDRSENTINEL not a number"),
    ("ZDRSENTINEL_bool", "ZDRSENTINEL not a boolean"),
    ("ZDRSENTINEL_obj", "{ZDRSENTINEL: broken object"),
    ("ZDRSENTINEL_odd", "ZDRSENTINEL custom type value"),
];

const CHILD_ENV: &str = "DYNAMO_PARSERS_ZDR_LOG_CHILD";

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

struct CaptureWriter(Arc<Mutex<Vec<u8>>>);

impl Write for CaptureWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Capture {
    type Writer = CaptureWriter;
    fn make_writer(&'a self) -> Self::Writer {
        CaptureWriter(self.0.clone())
    }
}

fn capture_logs(run: impl FnOnce()) -> String {
    let capture = Capture::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_writer(capture.clone())
        .with_ansi(false)
        .without_time()
        .finish();
    tracing::subscriber::with_default(subscriber, run);
    String::from_utf8_lossy(&capture.0.lock().unwrap()).into_owned()
}

fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "ZDRSENTINEL_str": {"type": "string"},
            "ZDRSENTINEL_int": {"type": "integer"},
            "ZDRSENTINEL_num": {"type": "number"},
            "ZDRSENTINEL_bool": {"type": "boolean"},
            "ZDRSENTINEL_obj": {"type": "object"},
            "ZDRSENTINEL_odd": {"type": "ZDRSENTINEL_customtype"},
        }
    })
}

fn tool(name: &str) -> Tool {
    Tool {
        name: name.to_string(),
        description: Some("ZDRSENTINEL description".to_string()),
        parameters: schema(),
        strict: None,
    }
}

/// No tools, the called tool declared, and only some other tool declared.
fn tool_sets() -> Vec<Vec<Tool>> {
    vec![vec![], vec![tool(NAME)], vec![tool(OTHER)]]
}

fn json_args() -> String {
    let map: serde_json::Map<String, serde_json::Value> = ARGS
        .iter()
        .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
        .collect();
    serde_json::Value::Object(map).to_string()
}

fn glm47_call(name: &str) -> String {
    let c = Glm47ParserConfig::default();
    let mut body = format!("{}{name}", c.tool_call_start);
    for (key, value) in ARGS {
        body.push_str(&format!(
            "{}{key}{}{}{value}{}",
            c.arg_key_start, c.arg_key_end, c.arg_value_start, c.arg_value_end
        ));
    }
    body + &c.tool_call_end
}

fn qwen_call(name: &str) -> String {
    let mut body = format!("<tool_call>\n<function={name}>\n");
    for (key, value) in ARGS {
        body.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
    }
    body + "</function>\n</tool_call>"
}

fn minimax_m2_call(name: &str) -> String {
    let mut body = format!("<minimax:tool_call>\n<invoke name=\"{name}\">\n");
    for (key, value) in ARGS {
        body.push_str(&format!("<parameter name=\"{key}\">{value}</parameter>\n"));
    }
    body + "</invoke>\n</minimax:tool_call>"
}

fn minimax_m3_call(name: &str) -> String {
    let c = MiniMaxM3ParserConfig::default();
    let ns = &c.namespace_token;
    let mut body = format!("{ns}<{}>\n{ns}<invoke name=\"{name}\">\n", c.tool_call_tag);
    for (key, value) in ARGS {
        body.push_str(&format!("{ns}<{key}>{value}{ns}</{key}>\n"));
    }
    body + &format!("{ns}</invoke>\n{ns}</{}>", c.tool_call_tag)
}

fn dsml_call(block: &str, sep: &str, name: &str) -> String {
    let mut body = format!("<｜DSML｜{sep}{block}>\n<｜DSML｜{sep}invoke name=\"{name}\">\n");
    for (key, value) in ARGS {
        body.push_str(&format!(
            "<｜DSML｜{sep}parameter name=\"{key}\" string=\"true\">{value}</｜DSML｜{sep}parameter>\n"
        ));
    }
    body + &format!("</｜DSML｜{sep}invoke>\n</｜DSML｜{sep}{block}>")
}

fn kimi_k2_calls(name: &str) -> Vec<String> {
    let c = KimiK2ParserConfig::default();
    vec![
        format!(
            "{}{}functions.{name}:0{}{}{}{}",
            c.section_start,
            c.call_start,
            c.argument_begin,
            json_args(),
            c.call_end,
            c.section_end
        ),
        format!(
            "{}{}functions.{name}:1{}{{ZDRSENTINEL broken json{}{}",
            c.section_start, c.call_start, c.argument_begin, c.call_end, c.section_end
        ),
    ]
}

fn kimi_k3_call(name: &str) -> String {
    let mut body = format!("<|open|>tools<|sep|><|open|>call tool=\"{name}\" index=\"1\"<|sep|>");
    for (key, value) in ARGS {
        body.push_str(&format!(
            "<|open|>argument key=\"{key}\" type=\"string\"<|sep|>{value}<|close|>argument<|sep|>"
        ));
    }
    body + "<|close|>call<|sep|><|close|>tools<|sep|>"
}

fn gemma4_calls(name: &str) -> Vec<String> {
    vec![
        format!(
            "<|tool_call>call:{name}{{ZDRSENTINEL_str:<|\"|>ZDRSENTINEL value<|\"|>,ZDRSENTINEL_int:7}}<tool_call|>"
        ),
        format!(
            "<|tool_call>call:{name}{{ZDRSENTINEL_key:ZDRSENTINEL_bare_value trailing}}<tool_call|>"
        ),
    ]
}

fn harmony_calls(name: &str) -> Vec<String> {
    vec![
        format!(
            "<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{}<|call|>",
            json_args()
        ),
        format!(
            "<|channel|>analysis<|message|>{THINKING}<|end|><|start|>assistant<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{}<|call|><|start|>assistant<|channel|>final<|message|>ZDRSENTINEL answer<|return|>",
            json_args()
        ),
        "<|channel|>ZDRSENTINELchannel<|message|>ZDRSENTINEL body<|end|>".to_string(),
    ]
}

fn hunyuan_call(name: &str) -> String {
    let mut body =
        format!("<tool_calls:opensource><tool_call:opensource>{name}<tool_sep:opensource>");
    for (key, value) in ARGS {
        body.push_str(&format!(
            "<arg_key:opensource>{key}</arg_key:opensource><arg_value:opensource>{value}</arg_value:opensource>"
        ));
    }
    body + "</tool_call:opensource></tool_calls:opensource>"
}

fn muse_glimmer_call(name: &str) -> String {
    let mut body = format!(
        "<|start|>assistant to=self<|message|>{THINKING}<|eom|><|start|>assistant to={name}<|message|><atem:function_calls>\n<atem:invoke name=\"{name}\">\n"
    );
    for (key, value) in ARGS {
        body.push_str(&format!(
            "<atem:parameter name=\"{key}\">{value}</atem:parameter>\n"
        ));
    }
    body + "</atem:invoke>\n</atem:function_calls><|eom|>"
}

/// Native wire format for a tool-call family. A family without a template
/// fails the test, so a new family cannot go unaudited.
fn tool_family_calls(family: &str, name: &str) -> Vec<String> {
    match family {
        "harmony" | "harmony_text" => harmony_calls(name),
        "deepseek_v4" => vec![dsml_call("tool_calls", "", name)],
        "qwen3_coder" => vec![qwen_call(name)],
        "muse_glimmer" => vec![muse_glimmer_call(name)],
        "minimax_m2" => vec![minimax_m2_call(name)],
        "minimax_m3" => vec![minimax_m3_call(name)],
        "gemma4" => gemma4_calls(name),
        "glm47" => vec![glm47_call(name)],
        "kimi_k2" => kimi_k2_calls(name),
        "kimi_k3" => vec![kimi_k3_call(name)],
        other => panic!("no ZDR log template for tool-call family {other}; add one"),
    }
}

/// Native wire format, with reasoning, for a unified family. A family without
/// a template fails the test.
fn unified_family_calls(family: &str, name: &str) -> Vec<String> {
    let canonical = canonical_unified_family(family).unwrap_or(family);
    let think = |call: String| format!("<think>{THINKING}</think>{call}");
    match canonical {
        "deepseek_v4" => vec![think(dsml_call("tool_calls", "", name))],
        "deepseek_v41" => vec![
            format!("{THINKING}</think>{}", dsml_call("calls", " ", name)),
            think(dsml_call("calls", " ", name)),
        ],
        "gemma4" => gemma4_calls(name)
            .into_iter()
            .map(|call| format!("<|channel>thought\n{THINKING}<channel|>{call}"))
            .collect(),
        "qwen3" => vec![think(qwen_call(name))],
        "muse_glimmer" => vec![muse_glimmer_call(name)],
        "kimi_k2" => kimi_k2_calls(name).into_iter().map(think).collect(),
        "kimi_k3" => vec![format!(
            "<|open|>think<|sep|>{THINKING}<|close|>think<|sep|>{}",
            kimi_k3_call(name)
        )],
        "hunyuan" => vec![format!(
            "<think:opensource>{THINKING}</think:opensource>{}",
            hunyuan_call(name)
        )],
        "mimo" => vec![think(qwen_call(name))],
        other => panic!("no ZDR log template for unified family {other}; add one"),
    }
}

/// Markers a family's calls open with, for the "opener lost" damage.
const OPENERS: &[&str] = &[
    "<tool_call>",
    "<|tool_call>",
    "<minimax:tool_call>",
    "]<]minimax[><tool_call>",
    "<|tool_calls_section_begin|>",
    "<|open|>tools<|sep|>",
    "<｜DSML｜tool_calls>",
    "<｜DSML｜ calls>",
    "<atem:function_calls>",
    "<|channel|>commentary",
    "<tool_calls:opensource>",
];

/// Markers a family's calls close with, for the "closer lost" and
/// "closer repeated" damage.
const CLOSERS: &[&str] = &[
    "</tool_call>",
    "<tool_call|>",
    "</minimax:tool_call>",
    "]<]minimax[></tool_call>",
    "<|tool_calls_section_end|>",
    "<|close|>tools<|sep|>",
    "</｜DSML｜tool_calls>",
    "</｜DSML｜ calls>",
    "</atem:function_calls>",
    "<|call|>",
    "</tool_calls:opensource>",
];

fn strip(text: &str, markers: &[&str]) -> String {
    markers
        .iter()
        .fold(text.to_string(), |acc, marker| acc.replace(marker, ""))
}

/// Whole and damaged messages, and separately every truncation of each call.
fn messages(calls: &[String], rogues: &[String]) -> (Vec<String>, Vec<String>) {
    let mut out = Vec::new();
    let mut truncations = Vec::new();
    for (call, rogue) in calls.iter().zip(rogues) {
        let closers: String = CLOSERS
            .iter()
            .filter(|closer| call.contains(**closer))
            .copied()
            .collect();
        out.push(format!("{BEFORE}{call}{AFTER}"));
        out.push(format!("{BEFORE}{call}{rogue}"));
        out.push(format!("{BEFORE}{call}{AFTER}{call}"));
        out.push(format!("{BEFORE}{}", strip(call, OPENERS)));
        out.push(format!("{BEFORE}{}{AFTER}", strip(call, CLOSERS)));
        out.push(format!(
            "{BEFORE}{}{AFTER}",
            strip(&strip(call, OPENERS), CLOSERS)
        ));
        out.push(format!("{BEFORE}{}", strip(rogue, OPENERS)));
        out.push(format!("{BEFORE}{call}{}", closers.repeat(3)));
        out.push(format!(
            "{BEFORE}{}{}",
            strip(call, OPENERS),
            closers.repeat(3)
        ));
        out.push(format!("{BEFORE}{}", call.replace(NAME, "")));
        out.push(format!(
            "{BEFORE}{}",
            strip(&call.replace(NAME, ""), CLOSERS)
        ));
        out.push(format!(
            "{BEFORE}{}",
            call.replace(NAME, "ZDRSENTINEL</arg_value>")
        ));
        let full = format!("{BEFORE}{call}");
        truncations.extend(
            full.char_indices()
                .map(|(i, _)| i)
                .filter(|&i| i > BEFORE.len())
                .map(|i| full[..i].to_string()),
        );
    }
    (out, truncations)
}

fn pieces(text: &str, size: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(size.max(1))
        .map(|piece| piece.iter().collect())
        .collect()
}

fn definitions(tools: &[Tool]) -> Vec<ToolDefinition> {
    tools.iter().map(ToolDefinition::from).collect()
}

fn run_tool_families(include_truncations: bool) {
    let sets = tool_sets();
    for family in REGISTERED_FAMILIES {
        let (whole, truncations) = messages(
            &tool_family_calls(family, NAME),
            &tool_family_calls(family, ROGUE),
        );
        for (i, message) in whole.iter().enumerate() {
            for (j, size) in [1, 7, usize::MAX].into_iter().enumerate() {
                let tools = &sets[(i + j) % sets.len()];
                tolerate_panic(|| {
                    let Ok(mut parser) = create_tool_parser_for_family(family, tools) else {
                        return;
                    };
                    for piece in pieces(message, size) {
                        let _ = parser.push(&piece);
                    }
                    let _ = parser.finish();
                });
            }
        }
        if include_truncations {
            for (message, tools) in truncations.iter().zip(sets.iter().cycle()) {
                tolerate_panic(|| {
                    if let Ok(mut parser) = create_tool_parser_for_family(family, tools) {
                        let _ = parser.parse_complete(message);
                    }
                });
            }
        }
        // Token-id input, for the families that take it.
        for ids in [
            &[TOKEN_SENTINEL, TOKEN_SENTINEL][..],
            &[BAD_TOKEN_SENTINEL],
            &[TOKEN_SENTINEL, BAD_TOKEN_SENTINEL],
        ] {
            if let Ok(mut parser) = create_tool_parser_for_family(family, &sets[1]) {
                let _ = parser.push_tokens(ids);
                let _ = parser.finish();
            }
        }
    }
}

fn native_init(starting_state: UnifiedParserStartingState) -> UnifiedParserInit {
    UnifiedParserInit {
        prompt_token_ids: vec![TOKEN_SENTINEL],
        starting_state,
        ..UnifiedParserInit::default()
    }
}

const STARTING_STATES: [UnifiedParserStartingState; 3] = [
    UnifiedParserStartingState::None,
    UnifiedParserStartingState::Reasoning,
    UnifiedParserStartingState::Response,
];

const POLICIES: [InvalidGuidedPayloadPolicy; 3] = [
    InvalidGuidedPayloadPolicy::Reject,
    InvalidGuidedPayloadPolicy::RecoverAsText,
    InvalidGuidedPayloadPolicy::StreamBestEffort,
];

/// Runs `drive`, tolerating a parser panic: this test judges only what
/// reaches the logs, and a panic is a defect of its own.
fn tolerate_panic(drive: impl FnOnce()) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(drive));
}

fn drive_unified(family: &str, tools: &[Tool], init: UnifiedParserInit, text: &str, size: usize) {
    tolerate_panic(|| {
        let Ok(mut parser) = create_unified_parser_for_family(family, tools) else {
            return;
        };
        if parser.initialize_request(init).is_err() {
            return;
        }
        let mut output = UnifiedParserOutput::default();
        for piece in pieces(text, size) {
            if parser.parse_into(&piece, &mut output).is_err() {
                return;
            }
        }
        if let Ok(mut rest) = parser.finish() {
            output.append(&mut rest);
        }
        // Callers collapse the stream with `assemble`, which parses arguments.
        let _ = assemble(&output.events);
    });
}

/// Guided payloads: valid, wrong shape, ambiguous, undeclared, truncated,
/// trailing bytes, markup only, and an envelope where bare arguments belong.
fn guided_payloads() -> Vec<String> {
    let args = json_args();
    vec![
        format!("[{{\"name\": \"{NAME}\", \"parameters\": {args}}}]"),
        format!("{{\"name\": \"{NAME}\", \"parameters\": {args}}}"),
        format!(
            "[{{\"name\": \"{NAME}\", \"parameters\": {args}, \"arguments\": {args}}}, {{\"name\": \"{ROGUE}\", \"parameters\": {args}}}]"
        ),
        format!("[{{\"name\": \"{ROGUE}\", \"parameters\": {args}}}, \"ZDRSENTINEL element\"]"),
        format!("[{{\"name\": \"{NAME}\", \"parameters\": {args}, \"arguments\": null}}]"),
        "{\"ZDRSENTINEL_k\": \"ZDRSENTINEL v\", \"ZDRSENTINEL_n\": 1234".to_string(),
        "{\"ZDRSENTINEL_k\": ZDRSENTINEL_bare} ZDRSENTINEL trailing".to_string(),
        format!("[{{\"name\": \"{NAME}\", \"parameters\": {{\"ZDRSENTINEL_k\": \"ZDRSENTINEL"),
        args.clone(),
        format!("{args} ZDRSENTINEL trailing"),
        format!("{{\"name\": \"{NAME}\", \"arguments\": {args}}}"),
        "\"ZDRSENTINEL bare string\"".to_string(),
        "ZDRSENTINEL not json".to_string(),
        "</tool_call>".to_string(),
        format!("<think>{THINKING}</think>{args}"),
    ]
}

fn run_unified_families(include_truncations: bool) {
    let sets = tool_sets();
    for family in REGISTERED_UNIFIED_FAMILIES {
        let (whole, truncations) = messages(
            &unified_family_calls(family, NAME),
            &unified_family_calls(family, ROGUE),
        );
        for (i, message) in whole.iter().enumerate() {
            for (j, size) in [1, 7, usize::MAX].into_iter().enumerate() {
                let tools = &sets[(i + j) % sets.len()];
                let state = STARTING_STATES[(i + j) % STARTING_STATES.len()];
                drive_unified(family, tools, native_init(state), message, size);
            }
        }
        if include_truncations {
            for (k, (message, tools)) in truncations.iter().zip(sets.iter().cycle()).enumerate() {
                let state = STARTING_STATES[k % STARTING_STATES.len()];
                drive_unified(family, tools, native_init(state), message, usize::MAX);
            }
        }
        for (i, payload) in guided_payloads().iter().enumerate() {
            for named_tool in [None, Some(OTHER.to_string()), Some(NAME.to_string())] {
                for policy in POLICIES {
                    for (j, size) in [1, 9, usize::MAX].into_iter().enumerate() {
                        let init = UnifiedParserInit {
                            prompt_token_ids: vec![TOKEN_SENTINEL],
                            starting_state: STARTING_STATES[(i + j) % STARTING_STATES.len()],
                            tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                                named_tool: named_tool.clone(),
                            },
                            invalid_guided_payload: policy,
                        };
                        drive_unified(family, &sets[(i + j) % sets.len()], init, payload, size);
                    }
                }
            }
        }
    }
}

/// The vendored batch parsers, called directly so EOF recovery runs both on
/// and off (the stream families fix it one way).
fn run_vendored_batch_parsers() {
    let sets = tool_sets();
    let glm_calls = [glm47_call(NAME)];
    let (glm, glm_truncations) = messages(&glm_calls, &[glm47_call(ROGUE)]);
    let (kimi, kimi_truncations) = messages(&kimi_k2_calls(NAME), &kimi_k2_calls(ROGUE));
    let (m3, m3_truncations) = messages(&[minimax_m3_call(NAME)], &[minimax_m3_call(ROGUE)]);
    let (qwen, qwen_truncations) = messages(&[qwen_call(NAME)], &[qwen_call(ROGUE)]);
    let (m2, m2_truncations) = messages(&[minimax_m2_call(NAME)], &[minimax_m2_call(ROGUE)]);
    let minimax_m2 = XmlParserConfig {
        tool_call_start_token: "<minimax:tool_call>".to_string(),
        tool_call_end_token: "</minimax:tool_call>".to_string(),
        function_start_token: "<invoke name=".to_string(),
        function_end_token: "</invoke>".to_string(),
        parameter_start_token: "<parameter name=".to_string(),
        parameter_end_token: "</parameter>".to_string(),
        ..XmlParserConfig::default()
    };
    for recovery in [false, true] {
        let glm_config = Glm47ParserConfig {
            allow_eof_recovery: recovery,
            ..Glm47ParserConfig::default()
        };
        let m3_config = MiniMaxM3ParserConfig {
            allow_eof_recovery: recovery,
            ..MiniMaxM3ParserConfig::default()
        };
        let qwen_config = XmlParserConfig {
            allow_eof_recovery: recovery,
            ..XmlParserConfig::default()
        };
        let m2_config = XmlParserConfig {
            allow_eof_recovery: recovery,
            ..minimax_m2.clone()
        };
        for tools in &sets {
            let defs = definitions(tools);
            let defs = Some(defs.as_slice());
            for message in glm.iter().chain(&glm_truncations) {
                let _ = try_tool_call_parse_glm47(message, &glm_config, defs);
            }
            for message in kimi.iter().chain(&kimi_truncations) {
                let _ = try_tool_call_parse_kimi_k2(message, &KimiK2ParserConfig::default(), defs);
            }
            for message in m3.iter().chain(&m3_truncations) {
                let _ = try_tool_call_parse_minimax_m3(message, &m3_config, defs);
            }
            for message in qwen.iter().chain(&qwen_truncations) {
                let _ = try_tool_call_parse_xml(message, &qwen_config, defs);
            }
            for message in m2.iter().chain(&m2_truncations) {
                let _ = try_tool_call_parse_xml(message, &m2_config, defs);
            }
        }
    }
}

/// Harmony's regex recovery with EOF recovery on; the stream family keeps it
/// off, so only a direct call reaches it.
fn run_harmony_regex_recovery() {
    for call in harmony_calls(NAME) {
        let call = std::slice::from_ref(&call);
        let (whole, truncations) = messages(call, call);
        for message in whole.iter().chain(&truncations) {
            for recovery in [false, true] {
                let _ = super::harmony_grammar::extract_calls_via_regex(message, recovery);
            }
        }
    }
}

fn run_corpus(include_truncations: bool) {
    run_tool_families(include_truncations);
    run_unified_families(include_truncations);
    if include_truncations {
        run_vendored_batch_parsers();
        run_harmony_regex_recovery();
    }
}

fn needles() -> [String; 3] {
    [
        SENTINEL.to_string(),
        TOKEN_SENTINEL.to_string(),
        BAD_TOKEN_SENTINEL.to_string(),
    ]
}

/// Messages of log sites that see request or model text. Each must fire in
/// the corpus, so a passing run proves those sites were exercised.
const REACHED: &[&str] = &[
    "GLM-4.7 parser dropping truncated tool_call block (recovery attempt failed)",
    "GLM-4.7 parser dropping truncated tool_call block (no end fence)",
    "GLM-4.7 parser dropping unparseable tool_call block",
    "GLM-4.7 parser dropping orphan tool-call marker tail",
    "GLM-4.7 parser dropping orphan close-marker spam after recovered bare call",
    "GLM-4.7 tool call references a function not in the request's tools list",
    "is not defined in the tools list (Gemma 4 parser)",
    "Failed to parse Gemma 4 args",
    "Failed to parse JSON arguments",
    "is not defined in the tools list.",
    "is not defined in the tool parameters",
    "is not an integer",
    "is not a float",
    "is not a boolean",
    "cannot be parsed with json.loads",
    "cannot be converted via Python `ast.literal_eval()`",
    "harmony decode pending token stream failed",
    "harmony decode failed while finishing token stream",
    "stripped harmony protocol content from normal_text",
    "recovered complete Harmony tool call at EOF",
    "emitted tool name does not match any registered tool",
    "tool-call arguments did not parse as JSON",
    "this call was streamed before it could be judged invalid",
    "bytes followed the named-choice argument object",
    "guided output contained no JSON payload",
    "guided output did not parse as a tool call",
    "named-choice payload carries `name`",
    "Hunyuan tool call names no offered tool",
    "MiMo tool call names an unknown tool",
];

#[test]
fn parser_logs_carry_no_request_or_model_text() {
    let logs = capture_logs(|| run_corpus(true));
    let report = leak_report(&logs, &needles());
    let missing: Vec<&str> = REACHED
        .iter()
        .copied()
        .filter(|message| !logs.contains(message))
        .collect();
    assert!(
        report.is_empty() && missing.is_empty(),
        "log sites that carried request or model text:\n{report}\n\
         sites that never fired: {missing:#?}"
    );
}

/// Re-runs the corpus in a child process with `DYNAMO_PARSERS_DEBUG=1` and
/// checks everything the debug wrappers (and anything else) wrote to stderr
/// and stdout.
#[test]
fn debug_stderr_carries_no_request_or_model_text() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "tool_calling::zdr_log_tests::debug_stderr_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .env(crate::DEBUG_ENV, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "child failed:\n{stdout}\n{stderr}");
    let report = leak_report(&format!("{stdout}\n{stderr}"), &needles());
    let missing: Vec<&str> = [
        "[dynamo-parsers-v2] family=",
        "call update(s)",
        "UNIFIED family=",
        "initialize prompt_token_ids_len=",
        "guided_json(named)",
        "delta(s)",
    ]
    .into_iter()
    .filter(|marker| !stderr.contains(marker))
    .collect();
    assert!(
        report.is_empty() && missing.is_empty(),
        "debug output that carried request or model text:\n{report}\n\
         debug lines that never appeared: {missing:#?}"
    );
}

/// The child half of [`debug_stderr_carries_no_request_or_model_text`]; a
/// no-op unless that test launched it.
#[test]
fn debug_stderr_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    run_corpus(false);
}

/// One entry per leaking log site: how many lines leaked, the site (level,
/// target and message, cut at the first sentinel), and the text around the
/// first sentinel.
fn leak_report(logs: &str, needles: &[String]) -> String {
    let mut sites: std::collections::BTreeMap<String, (usize, String)> = Default::default();
    for line in logs.lines() {
        let Some(at) = needles
            .iter()
            .filter_map(|needle| find_ignore_ascii_case(line, needle))
            .min()
        else {
            continue;
        };
        let site: String = line[..at].chars().take(140).collect();
        let from = floor_boundary(line, at.saturating_sub(60));
        let to = floor_boundary(line, (at + 60).min(line.len()));
        let entry = sites
            .entry(site)
            .or_insert_with(|| (0, line[from..to].to_string()));
        entry.0 += 1;
    }
    sites
        .iter()
        .map(|(site, (count, around))| format!("{count:>6} x {site}\n         ...{around}..."))
        .collect::<Vec<_>>()
        .join("\n")
}

fn find_ignore_ascii_case(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .as_bytes()
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
}

fn floor_boundary(text: &str, mut index: usize) -> usize {
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

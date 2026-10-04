// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parser logs must never carry request or model text.
//!
//! Deployments ship these logs to shared storage, and zero-data-retention
//! customers require that no prompt or completion text lands there. This test
//! drives every registered tool-call parser (batch, EOF recovery on and off,
//! and the streaming jail) and every registered reasoning parser over
//! well-formed, truncated and malformed inputs in which every content position
//! (prose, reasoning, tool names, argument keys and values, schema type names,
//! the named tool choice) carries a sentinel. It captures every event at TRACE
//! and fails if the sentinel, or a sentinel token id, appears in the output.

use std::io::Write;
use std::sync::{Arc, Mutex};

use dynamo_parsers::tool_calling::config::{JsonParserConfig, ParserConfig, XmlParserConfig};
use dynamo_parsers::tool_calling::jail::{Annotated, JailedStream, apply_tool_calling_jail};
use dynamo_parsers::tool_calling::json::JsonParserType;
use dynamo_parsers::tool_calling::parsers::{get_tool_parser_map, try_tool_call_parse};
use dynamo_parsers::{
    ReasoningParser, ReasoningParserType, ToolCallConfig, ToolDefinition,
    get_available_reasoning_parsers, try_tool_call_parse_aggregate,
    try_tool_call_parse_aggregate_finalize,
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    ChatCompletionToolChoiceOption, CreateChatCompletionStreamResponse, FinishReason, Role,
};
use futures::StreamExt;
use openai_harmony::{HarmonyEncodingName, load_harmony_encoding};
use tracing_subscriber::fmt::MakeWriter;

/// Every content position carries this marker. The check is case-insensitive.
const SENTINEL: &str = "ZDRSENTINEL";
/// A Harmony vocabulary id fed as model output; it must not appear in logs.
const TOKEN_SENTINEL: u32 = 187_345;

const NAME: &str = "ZDRSENTINEL_tool";
const ROGUE: &str = "ZDRSENTINEL_rogue";
const OTHER: &str = "ZDRSENTINEL_other";
const BEFORE: &str = "ZDRSENTINEL prose before. ";
const AFTER: &str = " ZDRSENTINEL prose after.";

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

fn tool_sets() -> Vec<Option<Vec<ToolDefinition>>> {
    let tool = |name: &str| ToolDefinition {
        name: name.to_string(),
        parameters: Some(schema()),
        strict: None,
    };
    vec![None, Some(vec![tool(NAME)]), Some(vec![tool(OTHER)])]
}

fn json_args() -> String {
    let map: serde_json::Map<String, serde_json::Value> = ARGS
        .iter()
        .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
        .collect();
    serde_json::Value::Object(map).to_string()
}

fn json_call(config: &JsonParserConfig, name: &str) -> String {
    let name_key = config.function_name_keys.first().map_or("name", |k| k);
    let args_key = config.arguments_keys.first().map_or("arguments", |k| k);
    format!(
        "{{\"{name_key}\": \"{name}\", \"{args_key}\": {}}}",
        json_args()
    )
}

fn xml_call(config: &XmlParserConfig, name: &str) -> String {
    let quote = |token: &str| if token.contains("name=") { "\"" } else { "" };
    let fq = quote(&config.function_start_token);
    let pq = quote(&config.parameter_start_token);
    let mut body = format!("{}{fq}{name}{fq}>\n", config.function_start_token);
    for (key, value) in ARGS {
        body.push_str(&format!(
            "{}{pq}{key}{pq}>\n{value}\n{}\n",
            config.parameter_start_token, config.parameter_end_token
        ));
    }
    body.push_str(&config.function_end_token);
    format!(
        "{}\n{body}\n{}",
        config.tool_call_start_token, config.tool_call_end_token
    )
}

/// One well-formed call per template, in the parser's own wire format, for
/// `name`. Built from the registered config wherever the config names the
/// markers.
fn calls_for(config: &ParserConfig, name: &str) -> Vec<String> {
    match config {
        ParserConfig::Json(c) => match c.parser_type {
            JsonParserType::Basic => {
                let call = json_call(c, name);
                let end = c.tool_call_end_tokens.first().cloned().unwrap_or_default();
                let mut out = vec![call.clone(), format!("[{call}]")];
                for start in &c.tool_call_start_tokens {
                    out.push(format!("{start}{call}{end}"));
                    out.push(format!("{start}[{call}, {}]{end}", json_call(c, ROGUE)));
                }
                out
            }
            JsonParserType::DeepseekV3 => vec![format!(
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                json_args()
            )],
            JsonParserType::DeepseekV31 => vec![format!(
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                json_args()
            )],
        },
        ParserConfig::Harmony(_) => vec![
            format!(
                "<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{}<|call|>",
                json_args()
            ),
            format!(
                "<|channel|>analysis<|message|>ZDRSENTINEL thinking<|end|><|start|>assistant<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{}<|call|><|start|>assistant<|channel|>final<|message|>ZDRSENTINEL answer<|return|>",
                json_args()
            ),
            "<|channel|>ZDRSENTINELchannel<|message|>ZDRSENTINEL body<|end|>".to_string(),
        ],
        ParserConfig::Pythonic => vec![format!(
            "[{name}(ZDRSENTINEL_str=\"ZDRSENTINEL value\", ZDRSENTINEL_int=7, ZDRSENTINEL_var=ZDRSENTINEL_name, **ZDRSENTINEL_kwargs)]"
        )],
        ParserConfig::Typescript => vec![],
        ParserConfig::Xml(c) => vec![xml_call(c, name)],
        ParserConfig::Dsml(c) => {
            let mut body = format!("{}\n{}\"{name}\">\n", c.block_start, c.invoke_start_prefix);
            for (key, value) in ARGS {
                body.push_str(&format!(
                    "{}\"{key}\" string=\"true\">{value}{}\n",
                    c.parameter_prefix, c.parameter_end
                ));
            }
            body.push_str(&format!("{}\n{}", c.invoke_end, c.block_end));
            vec![body]
        }
        ParserConfig::KimiK2(c) => vec![
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
        ],
        ParserConfig::KimiK3(_) => {
            let mut body =
                format!("<|open|>tools<|sep|><|open|>call tool=\"{name}\" index=\"1\"<|sep|>");
            for (key, value) in ARGS {
                body.push_str(&format!(
                    "<|open|>argument key=\"{key}\" type=\"string\"<|sep|>{value}<|close|>argument<|sep|>"
                ));
            }
            body.push_str("<|close|>call<|sep|><|close|>tools<|sep|>");
            vec![body]
        }
        ParserConfig::Glm47(c) => {
            let mut body = format!("{}{name}", c.tool_call_start);
            for (key, value) in ARGS {
                body.push_str(&format!(
                    "{}{key}{}{}{value}{}",
                    c.arg_key_start, c.arg_key_end, c.arg_value_start, c.arg_value_end
                ));
            }
            body.push_str(&c.tool_call_end);
            vec![body]
        }
        ParserConfig::MiniMaxM3(c) => {
            let ns = &c.namespace_token;
            let mut body = format!("{ns}<{}>\n{ns}<invoke name=\"{name}\">\n", c.tool_call_tag);
            for (key, value) in ARGS {
                body.push_str(&format!("{ns}<{key}>{value}{ns}</{key}>\n"));
            }
            body.push_str(&format!("{ns}</invoke>\n{ns}</{}>", c.tool_call_tag));
            vec![body]
        }
        ParserConfig::Gemma4 => {
            let start = &config.tool_call_start_tokens()[0];
            let end = &config.tool_call_end_tokens()[0];
            vec![
                format!(
                    "{start}call:{name}{{ZDRSENTINEL_str:<|\"|>ZDRSENTINEL value<|\"|>,ZDRSENTINEL_int:7}}{end}"
                ),
                format!(
                    "{start}call:{name}{{ZDRSENTINEL_key:ZDRSENTINEL_bare_value trailing}}{end}"
                ),
            ]
        }
        ParserConfig::Inkling(_) => {
            let starts = config.tool_call_start_tokens();
            let end = &config.tool_call_end_tokens()[0];
            vec![format!(
                "{}{name}{}{{\"name\":\"{name}\",\"args\":{}}}{end}",
                starts[0],
                starts[1],
                json_args()
            )]
        }
    }
}

/// The config with every EOF-recovery switch it has set to `on`. Production
/// leaves some of these off (GLM-4.7 always), so the registry alone never
/// reaches their recovery paths.
fn with_eof_recovery(config: &ParserConfig, on: bool) -> ToolCallConfig {
    let mut config = config.clone();
    match &mut config {
        ParserConfig::Json(c) | ParserConfig::Harmony(c) => c.allow_eof_recovery = on,
        ParserConfig::Xml(c) => c.allow_eof_recovery = on,
        ParserConfig::Dsml(c) => c.allow_eof_recovery = on,
        ParserConfig::Glm47(c) => c.allow_eof_recovery = on,
        ParserConfig::MiniMaxM3(c) => c.allow_eof_recovery = on,
        ParserConfig::KimiK3(c) => c.allow_eof_recovery = on,
        ParserConfig::Inkling(c) => c.allow_eof_recovery = on,
        ParserConfig::Pythonic
        | ParserConfig::Typescript
        | ParserConfig::KimiK2(_)
        | ParserConfig::Gemma4 => {}
    }
    ToolCallConfig {
        parser_config: config,
        structural_tag_builder: None,
    }
}

/// Well-formed messages with the structural damage the parsers have recovery
/// paths for, and separately every truncation of each well-formed call.
fn messages_for(config: &ParserConfig) -> (Vec<String>, Vec<String>) {
    let starts: Vec<String> = config
        .tool_call_start_tokens()
        .into_iter()
        .filter(|t| !t.is_empty())
        .collect();
    let ends: Vec<String> = config
        .tool_call_end_tokens()
        .into_iter()
        .filter(|t| !t.is_empty())
        .collect();
    let mut out = Vec::new();
    let mut truncations = Vec::new();
    for (call, rogue) in calls_for(config, NAME)
        .into_iter()
        .zip(calls_for(config, ROGUE))
    {
        let strip = |text: &str, tokens: &[String]| {
            tokens.iter().fold(text.to_string(), |acc, token| {
                acc.replace(token.as_str(), "")
            })
        };
        out.push(format!("{BEFORE}{call}{AFTER}"));
        out.push(format!("{BEFORE}{call}{rogue}"));
        out.push(format!("{BEFORE}{call}{AFTER}{call}"));
        out.push(format!("{BEFORE}{}", strip(&call, &starts)));
        out.push(format!("{BEFORE}{}{AFTER}", strip(&call, &ends)));
        out.push(format!(
            "{BEFORE}{}{AFTER}",
            strip(&strip(&call, &starts), &ends)
        ));
        out.push(format!("{BEFORE}{}", strip(&rogue, &starts)));
        out.push(format!("{BEFORE}{call}{}", ends.concat().repeat(3)));
        out.push(format!(
            "{BEFORE}{}{}",
            strip(&call, &starts),
            ends.concat().repeat(3)
        ));
        out.push(format!("{BEFORE}{}", call.replace(NAME, "")));
        out.push(format!("{BEFORE}{}", strip(&call.replace(NAME, ""), &ends)));
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

fn chunk(
    content: String,
    finish: Option<FinishReason>,
) -> Annotated<CreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: Some(ChatCompletionMessageContent::Text(content)),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: finish,
        logprobs: None,
    };
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: "zdr".to_string(),
            choices: vec![choice],
            created: 0,
            model: "zdr".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

fn chunks(text: &str, size: usize) -> Vec<Annotated<CreateChatCompletionStreamResponse>> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<_> = chars
        .chunks(size.max(1))
        .map(|c| chunk(c.iter().collect(), None))
        .collect();
    out.push(chunk(String::new(), Some(FinishReason::Stop)));
    out
}

async fn drain(stream: impl futures::Stream<Item = Annotated<CreateChatCompletionStreamResponse>>) {
    let _ = stream.collect::<Vec<_>>().await;
}

async fn run_tool_parsers() -> usize {
    let mut runs = 0;
    let mut parsers: Vec<_> = get_tool_parser_map().iter().collect();
    parsers.sort_by_key(|(name, _)| **name);
    for (name, config) in parsers {
        let (messages, truncations) = messages_for(&config.parser_config);
        let recovering = with_eof_recovery(&config.parser_config, true);
        let tool_sets = tool_sets();
        // Whole and damaged messages run against every tool set; truncations
        // rotate through them.
        let runs_for = messages
            .iter()
            .flat_map(|message| tool_sets.iter().map(move |tools| (message, tools)))
            .chain(truncations.iter().zip(tool_sets.iter().cycle()));
        for (message, tools) in runs_for {
            let tools = tools.as_deref();
            let _ = try_tool_call_parse_aggregate(message, Some(name), tools).await;
            let _ = try_tool_call_parse_aggregate_finalize(message, Some(name), tools).await;
            let _ = try_tool_call_parse(message, &recovering, tools).await;
            runs += 3;
        }
        // The streaming jail, as the serving layer drives it. Truncations are
        // covered above; here the whole and damaged messages stream in pieces.
        for (i, message) in messages.iter().enumerate() {
            for (j, size) in [1, 7, usize::MAX].into_iter().enumerate() {
                let tools = &tool_sets[(i + j) % tool_sets.len()];
                let jail = JailedStream::builder()
                    .tool_call_parser(name.to_string())
                    .named_tool_filter(OTHER);
                let jail = match tools.clone() {
                    Some(tools) => jail.tool_definitions(tools),
                    None => jail,
                };
                drain(
                    jail.build()
                        .apply(futures::stream::iter(chunks(message, size))),
                )
                .await;
                drain(apply_tool_calling_jail(
                    Some(name.to_string()),
                    None,
                    tools.clone(),
                    false,
                    futures::stream::iter(chunks(message, size)),
                ))
                .await;
                runs += 2;
            }
        }
    }
    runs
}

async fn run_guided_jail() {
    let named = || ChatCompletionToolChoiceOption::Named(OTHER.to_string().into());
    let payloads = [
        format!(
            "[{{\"name\": \"{NAME}\", \"parameters\": {}}}]",
            json_args()
        ),
        json_args(),
        format!("{{\"name\": \"{NAME}\", \"arguments\": {}}}", json_args()),
        "ZDRSENTINEL not json".to_string(),
        format!("[{{\"name\": \"{NAME}\", \"parameters\": {{\"ZDRSENTINEL_k\": \"ZDRSENTINEL"),
    ];
    for payload in &payloads {
        for choice in [
            Some(ChatCompletionToolChoiceOption::Required),
            Some(named()),
        ] {
            for parser in [None, Some("hermes".to_string()), Some("glm47".to_string())] {
                for size in [1, 7, usize::MAX] {
                    for tools in tool_sets() {
                        drain(apply_tool_calling_jail(
                            parser.clone(),
                            choice.clone(),
                            tools,
                            false,
                            futures::stream::iter(chunks(payload, size)),
                        ))
                        .await;
                    }
                }
            }
        }
    }
}

fn run_reasoning_parsers() {
    let texts = [
        "<think>ZDRSENTINEL thinking</think>ZDRSENTINEL answer".to_string(),
        "ZDRSENTINEL thinking</think>ZDRSENTINEL answer".to_string(),
        "[THINK]ZDRSENTINEL thinking[/THINK]ZDRSENTINEL answer".to_string(),
        "<mm:think>ZDRSENTINEL thinking</mm:think>ZDRSENTINEL answer".to_string(),
        "<|channel>thought\nZDRSENTINEL thinking<channel|>ZDRSENTINEL answer".to_string(),
        "<|open|>think<|sep|>ZDRSENTINEL thinking<|close|>think<|sep|>ZDRSENTINEL answer"
            .to_string(),
        "Here is my thought process: ZDRSENTINEL thinking Here is my response: ZDRSENTINEL answer"
            .to_string(),
        "<|channel|>analysis<|message|>ZDRSENTINEL thinking<|end|><|start|>assistant<|channel|>final<|message|>ZDRSENTINEL answer<|return|>"
            .to_string(),
        format!(
            "<|channel|>analysis<|message|>ZDRSENTINEL thinking<|end|><|start|>assistant<|channel|>commentary to=functions.{NAME} <|constrain|>json<|message|>{}<|call|>",
            json_args()
        ),
        "<|channel|>ZDRSENTINELchannel<|message|>ZDRSENTINEL body".to_string(),
        "<|start|>ZDRSENTINELrole<|channel|>final<|message|>ZDRSENTINEL answer<|end|>".to_string(),
    ];
    // Token-id input: a complete message, then a token where only <|start|>
    // is legal, so the Harmony parser rejects the sentinel id.
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let mut rejected = encoding
        .tokenizer()
        .encode_with_special_tokens("<|channel|>final<|message|>ZDRSENTINEL answer<|end|>");
    rejected.push(TOKEN_SENTINEL);
    let token_runs: [&[u32]; 3] = [
        &rejected,
        &[TOKEN_SENTINEL, TOKEN_SENTINEL],
        &[TOKEN_SENTINEL],
    ];
    let mut names = get_available_reasoning_parsers();
    names.sort();
    for name in names {
        for text in &texts {
            for in_reasoning in [false, true] {
                let mut parser = ReasoningParserType::get_reasoning_parser_from_name(name);
                parser.set_in_reasoning(in_reasoning);
                let _ = parser.detect_and_parse_reasoning(text, &[]);

                for size in [1, 5, 13] {
                    let mut parser = ReasoningParserType::get_reasoning_parser_from_name(name);
                    parser.set_in_reasoning(in_reasoning);
                    let chars: Vec<char> = text.chars().collect();
                    for piece in chars.chunks(size) {
                        let piece: String = piece.iter().collect();
                        let _ = parser.parse_reasoning_streaming_incremental(&piece, &[]);
                    }
                    let _ = parser.finish_reasoning_stream();
                }
            }
        }
        for ids in token_runs {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name(name);
            let _ = parser.detect_and_parse_reasoning("", ids);
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name(name);
            let _ = parser.parse_reasoning_streaming_incremental("", ids);
            let _ = parser.finish_reasoning_stream();
        }
    }
}

/// Messages of log sites that see request or model text. Each must fire in
/// the corpus, so a passing run proves those sites were exercised.
const REACHED: &[&str] = &[
    "GLM-4.7 parser dropping truncated tool_call block (recovery attempt failed)",
    "GLM-4.7 parser dropping truncated tool_call block (no end fence)",
    "GLM-4.7 parser dropping unparseable tool_call block",
    "GLM-4.7 parser dropping orphan tool-call marker tail",
    "GLM-4.7 parser dropping orphan close-marker spam after recovered bare call",
    "tool-call parser errored; dropping buffered content to avoid marker leak",
    "tool_choice=named: parser emitted no matching tool calls",
    "tool_choice=named: parsers emitted no matching tool calls",
    "Creating MarkerMatcher",
    "holding",
    "DSML strip (recovery)",
    "DSML strip (success)",
    "is not defined in the tools list (Gemma 4 parser)",
    "Failed to parse Gemma 4 args",
    "gemma4 strip (recovery)",
    "gemma4 strip (success)",
    "stripped harmony protocol content from normal_text",
    "Failed to parse messages from completion tokens",
    "Dropping unparseable tool-call content",
    "Skipping **kwargs in pythonic tool call",
    "Skipping non-constant argument",
    "Failed to parse JSON arguments",
    "is not defined in the tools list.",
    "is not defined in the tool parameters",
    "is not an integer",
    "is not a float",
    "is not a boolean",
    "cannot be parsed with json.loads",
    "cannot be converted via Python `ast.literal_eval()`",
    "Harmony parse error",
    "Processing token",
    "Processing streaming token",
    "Shouldn't be delta content after in channel",
];

#[test]
fn parser_logs_carry_no_request_or_model_text() {
    let mut runs = 0;
    let logs = capture_logs(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            runs = run_tool_parsers().await;
            run_guided_jail().await;
        });
        run_reasoning_parsers();
    });
    assert!(runs > 0);

    let report = leak_report(&logs, &[SENTINEL, &TOKEN_SENTINEL.to_string()]);
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

/// One entry per leaking log site: how many lines leaked, the site (level,
/// target and message, cut at the first sentinel), and the text around the
/// first sentinel.
fn leak_report(logs: &str, needles: &[&str]) -> String {
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

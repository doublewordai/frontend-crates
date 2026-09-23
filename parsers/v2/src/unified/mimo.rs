// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified parser for XiaomiMiMo MiMo output.
//!
//! ```text
//! reasoning:  <think>…</think>
//! tool calls: <tool_call>
//!             <function=NAME>
//!             <parameter=KEY>VALUE</parameter>
//!             </function>
//!             </tool_call>
//! ```
//!
//! The markup is Qwen3-Coder's, so the scan core, reasoning channel and guided
//! prefix policy are the Qwen3 ones. Value decoding differs: MiMo's prompt states
//! that the text between the parameter tags is the value exactly, including
//! leading and trailing whitespace, and its reference parser does no trimming.
//! The Qwen3-Coder emitter trims every value, which rewrites a patch or a file
//! body, so MiMo types each value itself: HTML references decoded as the
//! reference parser's `html.unescape` does, every other byte kept, and the type
//! taken from the tool schema.

use serde_json::{Map, Value};

use crate::tool_calling::qwen3_coder::spec as qwen3_coder_spec;
use crate::tool_calling::scan::{
    GuidedInvokePrefix, GuidedInvokePrefixContext, InvokeBoundary, InvokeBoundaryFactory,
    InvokeEmitter, ReasoningSpec, WrappedBlockScanner, WrappedBlockSpec,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::unified::qwen3::qwen_guided_prefix;
use crate::unified::{
    GuidedPrefix, GuidedPrefixContext, GuidedPrefixScanner, GuidedRouted, ScannerUnified,
    UnifiedParser,
};

const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

fn find_from(text: &str, start: usize, marker: &str) -> Option<usize> {
    text[start..].find(marker).map(|at| start + at)
}

/// Where a scan may resume without missing a marker split across the append.
fn next_scan_start(text: &str, marker_len: usize) -> usize {
    let mut start = text.len().saturating_sub(marker_len.saturating_sub(1));
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    start
}

/// Where the scan sits inside one `<function=…>` invoke.
#[derive(Default)]
enum InvokePosition {
    /// Between parameters, where `</function>` ends the invoke.
    #[default]
    Between,
    /// Inside a `<parameter=` header, before its `>`.
    Header,
    /// Inside a parameter value, which only `</parameter>` ends.
    Value,
}

/// Finds the end of one invoke while skipping parameter values, so a value
/// that spells `</function>` stays part of the value. A `<function=` header
/// ahead of guided JSON is classified by the Qwen3 guided prefix rules.
struct MimoInvocationBoundary {
    position: InvokePosition,
    scan_from: usize,
    guided_prefix: Box<dyn GuidedPrefixScanner>,
}

impl Default for MimoInvocationBoundary {
    fn default() -> Self {
        Self {
            position: InvokePosition::default(),
            scan_from: 0,
            guided_prefix: qwen_guided_prefix(),
        }
    }
}

impl InvokeBoundary for MimoInvocationBoundary {
    fn owns_guided_prefix(&self) -> bool {
        true
    }

    fn guided_prefix_append(
        &mut self,
        candidate: &str,
        append: &str,
        context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        let prefix = self.guided_prefix.append(
            candidate,
            append,
            GuidedPrefixContext {
                text: candidate,
                at: 0,
                outside_reasoning: context.outside_reasoning,
                payload_is_empty: context.payload_is_empty,
                followed_by_competing_marker: context.followed_by_competing_marker,
            },
        );
        Some(match prefix {
            // A bare `<function=` straight into the payload names no function, so
            // it is framing, not a native call.
            GuidedPrefix::NoMatch
                if candidate
                    .get(FUNCTION_OPEN.len()..)
                    .is_some_and(|header| header.starts_with(['{', '['])) =>
            {
                GuidedInvokePrefix::Match(FUNCTION_OPEN.len())
            }
            GuidedPrefix::NoMatch => GuidedInvokePrefix::NoMatch,
            GuidedPrefix::Pending => GuidedInvokePrefix::Pending,
            GuidedPrefix::Strip(len) => GuidedInvokePrefix::Strip(len),
            // Everything before the JSON payload is header.
            GuidedPrefix::Match => GuidedInvokePrefix::Match(
                candidate
                    .find(['{', '['])
                    .filter(|at| *at >= FUNCTION_OPEN.len())
                    .unwrap_or(FUNCTION_OPEN.len()),
            ),
        })
    }

    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        _flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        loop {
            match self.position {
                InvokePosition::Between => {
                    let close = find_from(candidate, self.scan_from, FUNCTION_CLOSE);
                    let parameter = find_from(candidate, self.scan_from, PARAMETER_OPEN);
                    match (close, parameter) {
                        (Some(close), Some(parameter)) if close < parameter => {
                            return Some(close + FUNCTION_CLOSE.len());
                        }
                        (_, Some(parameter)) => {
                            self.position = InvokePosition::Header;
                            self.scan_from = parameter + PARAMETER_OPEN.len();
                        }
                        (Some(close), None) => return Some(close + FUNCTION_CLOSE.len()),
                        (None, None) => {
                            self.scan_from = next_scan_start(
                                candidate,
                                FUNCTION_CLOSE.len().max(PARAMETER_OPEN.len()),
                            );
                            return None;
                        }
                    }
                }
                InvokePosition::Header => {
                    let Some(header_end) = find_from(candidate, self.scan_from, ">") else {
                        self.scan_from = candidate.len();
                        return None;
                    };
                    self.position = InvokePosition::Value;
                    self.scan_from = header_end + 1;
                }
                InvokePosition::Value => {
                    let Some(end) = find_from(candidate, self.scan_from, PARAMETER_CLOSE) else {
                        self.scan_from = next_scan_start(candidate, PARAMETER_CLOSE.len());
                        return None;
                    };
                    self.position = InvokePosition::Between;
                    self.scan_from = end + PARAMETER_CLOSE.len();
                }
            }
        }
    }

    fn opens(&self, _text: &str, _at: usize) -> bool {
        true
    }

    fn holdback(&self, _text: &str) -> usize {
        0
    }

    fn resync(&mut self, _text: &str, _flush: bool, _tool_index: usize) -> Option<usize> {
        None
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn invocation_boundary() -> Box<dyn InvokeBoundary> {
    Box::new(MimoInvocationBoundary::default())
}

/// HTML character reference decoding with the HTML5 rules the engine parser's
/// `html.unescape` applies to every value before typing it: named references with
/// and without a semicolon, and numeric references remapped the way HTML5 remaps
/// them. Everything that is not a reference, whitespace included, is left untouched.
fn html_unescape(value: &str) -> String {
    htmlize::unescape(value).into_owned()
}

/// The JSON Schema `type` declared for one parameter, as the reference parser reads
/// it: a string names the type, any other `type` value (a nullable union such as
/// `["integer", "null"]`) is spelled out and so falls through to JSON parsing, and a
/// schema with no `type` is a string.
fn schema_type(tools: &[Tool], function: &str, parameter: &str) -> String {
    match tools
        .iter()
        .find(|tool| tool.name == function)
        .and_then(|tool| tool.parameters.get("properties"))
        .and_then(|properties| properties.get(parameter))
        .and_then(|schema| schema.get("type"))
    {
        Some(Value::String(declared)) => declared.clone(),
        Some(declared) => declared.to_string(),
        None => "string".to_string(),
    }
}

/// Type one raw parameter value against its schema.
///
/// A string-typed value is the model's bytes verbatim. A value that does not parse
/// as its declared type stays a string rather than failing the call, which is what
/// the engine's parser does.
fn typed_value(raw: &str, declared: &str) -> serde_json::Value {
    let value = html_unescape(raw);
    if value.eq_ignore_ascii_case("null") {
        return serde_json::Value::Null;
    }
    let as_string = || serde_json::Value::String(value.clone());
    let trimmed = value.trim();
    let declared = declared.to_ascii_lowercase();
    match declared.as_str() {
        "string" | "str" | "text" | "varchar" | "char" | "enum" => as_string(),
        // The reference converter compares the untrimmed value, and anything
        // other than `true` is false.
        "boolean" | "bool" | "binary" => {
            serde_json::Value::Bool(value.eq_ignore_ascii_case("true"))
        }
        _ if declared.starts_with("int")
            || declared.starts_with("uint")
            || declared.starts_with("long")
            || declared.starts_with("short")
            || declared.starts_with("unsigned") =>
        {
            trimmed
                .parse::<i64>()
                .map(serde_json::Value::from)
                .or_else(|_| trimmed.parse::<u64>().map(serde_json::Value::from))
                .unwrap_or_else(|_| as_string())
        }
        _ if declared.starts_with("num") || declared.starts_with("float") => trimmed
            .parse::<f64>()
            .ok()
            .filter(|number| number.is_finite())
            .map(|number| {
                if number.fract() == 0.0 && number.abs() < i64::MAX as f64 {
                    serde_json::Value::from(number as i64)
                } else {
                    serde_json::Value::from(number)
                }
            })
            .unwrap_or_else(as_string),
        _ => serde_json::from_str(trimmed).unwrap_or_else(|_| as_string()),
    }
}

/// Decode one `<function=…>…</function>` invoke into a call, or `None` when it is
/// not a complete envelope. Parameters are read in order and each value runs to
/// its own `</parameter>`, so a value may spell any other marker.
fn decode_invoke(invoke: &str, tools: &[Tool]) -> Option<(String, Map<String, Value>)> {
    let (name, mut rest) = invoke.strip_prefix(FUNCTION_OPEN)?.split_once('>')?;
    let name = name.trim().to_string();
    let mut arguments = Map::new();
    loop {
        let parameter = rest.find(PARAMETER_OPEN);
        let close = rest.find(FUNCTION_CLOSE);
        match (parameter, close) {
            (Some(parameter), close) if close.is_none_or(|close| parameter < close) => {
                let (key, value) = rest[parameter + PARAMETER_OPEN.len()..].split_once('>')?;
                let value_len = value.find(PARAMETER_CLOSE)?;
                let key = key.trim().to_string();
                let declared = schema_type(tools, &name, &key);
                arguments.insert(key, typed_value(&value[..value_len], &declared));
                rest = &value[value_len + PARAMETER_CLOSE.len()..];
            }
            (_, Some(_)) => return Some((name, arguments)),
            _ => return None,
        }
    }
}

/// Types each complete `<function=…>` invoke the Qwen3-Coder scan core delimits.
struct MimoEmitter {
    tools: Vec<Tool>,
}

impl InvokeEmitter for MimoEmitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let Some((name, arguments)) = decode_invoke(invoke, &self.tools) else {
            tracing::warn!(
                why = "mimo_unparsable_invoke",
                "MiMo tool-call invoke decoded to no call"
            );
            return Ok(None);
        };
        if !self.tools.is_empty() && !self.tools.iter().any(|tool| tool.name == name) {
            tracing::warn!(name, "MiMo tool call names an unknown tool");
            return Ok(None);
        }
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(name),
            arguments: Value::Object(arguments).to_string(),
            complete: true,
        }))
    }
}

/// Build the MiMo unified parser for one stream.
pub(crate) fn mimo_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    let spec = WrappedBlockSpec {
        family: "mimo",
        invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(invocation_boundary)),
        ..qwen3_coder_spec()
    };
    let scanner = WrappedBlockScanner::new(
        spec,
        MimoEmitter {
            tools: tools.to_vec(),
        },
    )
    .with_reasoning(ReasoningSpec {
        start: "<think>",
        end: "</think>",
        ..Default::default()
    });
    Box::new(GuidedRouted::new(ScannerUnified::new(scanner)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserInit, UnifiedParserOutput,
        UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
    };
    use serde_json::json;

    fn tools() -> Vec<Tool> {
        let tool = |name: &str, properties: serde_json::Value| Tool {
            name: name.to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": properties}),
            strict: None,
        };
        vec![
            tool("execute_bash", json!({"command": {"type": "string"}})),
            tool(
                "edit_file",
                json!({"path": {"type": "string"}, "body": {"type": "string"}}),
            ),
            tool(
                "get_weather",
                json!({
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "metric": {"type": "boolean"},
                    "tags": {"type": "array"}
                }),
            ),
        ]
    }

    fn run(
        chunks: &[&str],
        starting_state: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
    ) -> Vec<UnifiedEvent> {
        let mut parser = mimo_unified(&tools());
        parser
            .initialize_request(UnifiedParserInit {
                starting_state,
                tool_output_mode,
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            parser.parse_into(chunk, &mut output).unwrap();
        }
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assemble(&output.events)
    }

    fn native(chunks: &[&str]) -> Vec<UnifiedEvent> {
        run(
            chunks,
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::Native,
        )
    }

    fn text(text: &str) -> UnifiedEvent {
        UnifiedEvent::Text { text: text.into() }
    }

    fn reasoning(text: &str) -> UnifiedEvent {
        UnifiedEvent::Reasoning { text: text.into() }
    }

    fn call(name: &str, arguments: serde_json::Value) -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: name.into(),
            arguments,
        }
    }

    fn assert_split_invariant(input: &str, starting_state: UnifiedParserStartingState) {
        let whole = run(&[input], starting_state, UnifiedToolOutputMode::Native);
        for (at, _) in input.char_indices().skip(1) {
            assert_eq!(
                run(
                    &[&input[..at], &input[at..]],
                    starting_state,
                    UnifiedToolOutputMode::Native
                ),
                whole,
                "split at byte {at} of {input:?}"
            );
        }
        let chars: Vec<String> = input.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(
            run(&chars, starting_state, UnifiedToolOutputMode::Native),
            whole,
            "char-at-a-time {input:?}"
        );
    }

    const BASH_CALL: &str = "<tool_call>\n<function=execute_bash>\n\
         <parameter=command>pwd && ls</parameter>\n</function>\n</tool_call>";

    #[test]
    fn plain_text_passes_through() {
        assert_eq!(
            native(&["This is a plain response."]),
            vec![text("This is a plain response.")]
        );
        assert_eq!(native(&["a < b and x<y>"]), vec![text("a < b and x<y>")]);
    }

    #[test]
    fn reasoning_block_then_answer() {
        assert_eq!(
            native(&["<think>Let me think.</think>The answer is 42."]),
            vec![reasoning("Let me think."), text("The answer is 42.")]
        );
    }

    #[test]
    fn prompt_opened_reasoning_closes_without_an_opener() {
        assert_eq!(
            run(
                &["Let me think.</think>The answer is 42."],
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::Native
            ),
            vec![reasoning("Let me think."), text("The answer is 42.")]
        );
    }

    #[test]
    fn single_tool_call() {
        assert_eq!(
            native(&[BASH_CALL]),
            vec![call("execute_bash", json!({"command": "pwd && ls"}))]
        );
    }

    #[test]
    fn text_reasoning_and_multiple_calls_stay_in_order() {
        let input = format!(
            "<think>Two things.</think>On it.{BASH_CALL}\n\
             <tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n\
             <parameter=days>3</parameter>\n</function>\n</tool_call>Done."
        );
        assert_eq!(
            native(&[&input]),
            vec![
                reasoning("Two things."),
                text("On it."),
                call("execute_bash", json!({"command": "pwd && ls"})),
                text("\n"),
                call("get_weather", json!({"city": "Paris", "days": 3})),
                text("Done."),
            ]
        );
    }

    #[test]
    fn parameter_values_are_preserved_byte_for_byte() {
        let body = "\nline one\n  indented\n\n";
        let input = format!(
            "<tool_call>\n<function=edit_file>\n<parameter=path>src/main.rs</parameter>\n\
             <parameter=body>{body}</parameter>\n</function>\n</tool_call>"
        );
        assert_eq!(
            native(&[&input]),
            vec![call(
                "edit_file",
                json!({"path": "src/main.rs", "body": body})
            )]
        );
    }

    #[test]
    fn values_are_typed_from_the_schema_and_entities_decoded() {
        let input = "<tool_call><function=get_weather>\
             <parameter=city>a &lt;b&gt; &amp; c</parameter>\
             <parameter=days>3</parameter>\
             <parameter=ratio>1.5</parameter>\
             <parameter=metric>true</parameter>\
             <parameter=tags>[\"a\", \"b\"]</parameter>\
             </function></tool_call>";
        assert_eq!(
            native(&[input]),
            vec![call(
                "get_weather",
                json!({
                    "city": "a <b> & c",
                    "days": 3,
                    "ratio": 1.5,
                    "metric": true,
                    "tags": ["a", "b"]
                })
            )]
        );
    }

    #[test]
    fn a_value_that_defies_its_type_stays_a_string() {
        let input = "<tool_call><function=get_weather>\
             <parameter=days>soon</parameter></function></tool_call>";
        assert_eq!(
            native(&[input]),
            vec![call("get_weather", json!({"days": "soon"}))]
        );
    }

    /// The reference converter compares the untrimmed value with `true`.
    #[test]
    fn booleans_follow_the_reference_converter() {
        let input = "<tool_call><function=get_weather>\
             <parameter=metric>maybe</parameter></function></tool_call>\
             <tool_call><function=get_weather>\
             <parameter=metric> true </parameter></function></tool_call>\
             <tool_call><function=get_weather>\
             <parameter=metric>TRUE</parameter></function></tool_call>";
        assert_eq!(
            native(&[input]),
            vec![
                call("get_weather", json!({"metric": false})),
                call("get_weather", json!({"metric": false})),
                call("get_weather", json!({"metric": true})),
            ]
        );
    }

    #[test]
    fn markup_inside_a_value_is_value_text() {
        let input = "<tool_call><function=execute_bash>\
             <parameter=command>echo '<think>' && echo '</tool_call'</parameter>\
             </function></tool_call>";
        assert_eq!(
            native(&[input]),
            vec![call(
                "execute_bash",
                json!({"command": "echo '<think>' && echo '</tool_call'"})
            )]
        );
    }

    #[test]
    fn a_call_opening_inside_a_thought_ends_it() {
        assert_eq!(
            run(
                &[&format!("Need the listing.{BASH_CALL}")],
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::Native
            ),
            vec![
                reasoning("Need the listing."),
                call("execute_bash", json!({"command": "pwd && ls"}))
            ]
        );
    }

    #[test]
    fn chunk_boundaries_never_change_the_result() {
        assert_split_invariant(
            &format!("<think>Think.</think>Text. {BASH_CALL} tail"),
            UnifiedParserStartingState::None,
        );
        assert_split_invariant(
            "thought</think>a < b, <b>bold</b>",
            UnifiedParserStartingState::Reasoning,
        );
    }

    #[test]
    fn split_marker_is_held_until_decided() {
        let mut parser = mimo_unified(&tools());
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("Hello <tool_c", &mut output).unwrap();
        assert_eq!(assemble(&output.events), vec![text("Hello ")]);
        parser.parse_into("ase is closed", &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(
            assemble(&output.events),
            vec![text("Hello <tool_case is closed")]
        );
    }

    #[test]
    fn unterminated_reasoning_is_reasoning() {
        assert_eq!(
            native(&["<think>still thinking when the budget ran out"]),
            vec![reasoning("still thinking when the budget ran out")]
        );
    }

    #[test]
    fn unterminated_call_is_dropped_without_leaking_markup() {
        assert_eq!(
            native(&["On it.<tool_call>\n<function=execute_bash>\n<parameter=command>pw"]),
            vec![text("On it.")]
        );
    }

    #[test]
    fn a_block_with_no_function_envelope_emits_nothing() {
        assert_eq!(native(&["<tool_call>garbage</tool_call>"]), vec![]);
    }

    #[test]
    fn unknown_tool_is_not_a_call() {
        let input = format!("<tool_call><function=nonexistent></function></tool_call>{BASH_CALL}");
        assert_eq!(
            native(&[&input]),
            vec![call("execute_bash", json!({"command": "pwd && ls"}))]
        );
    }

    #[test]
    fn guided_named_and_required_choices() {
        assert_eq!(
            run(
                &["Pick Paris.</think>{\"city\": ", "\"Paris\"}"],
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("get_weather".into())
                }
            ),
            vec![
                reasoning("Pick Paris."),
                call("get_weather", json!({"city": "Paris"}))
            ]
        );
        assert_eq!(
            run(
                &[r#"[{"name":"execute_bash","arguments":{"command":"ls"}}]"#],
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None }
            ),
            vec![call("execute_bash", json!({"command": "ls"}))]
        );
        assert_eq!(
            run(
                &[r#"[{"name":"execute_bash","argu"#],
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None }
            ),
            vec![text(r#"[{"name":"execute_bash","argu"#)]
        );
    }

    #[test]
    fn guided_payload_after_a_generated_thought() {
        let mode = || UnifiedToolOutputMode::GuidedJson {
            named_tool: Some("get_weather".into()),
        };
        let input = "\n<think>Pick a city.</think>\n{\"city\":\"Paris </think> <think>\"}";
        let expected = vec![
            reasoning("Pick a city."),
            call("get_weather", json!({"city": "Paris </think> <think>"})),
        ];
        assert_eq!(
            run(&[input], UnifiedParserStartingState::None, mode()),
            expected
        );
        for at in 1..input.len() {
            assert_eq!(
                run(
                    &[&input[..at], &input[at..]],
                    UnifiedParserStartingState::None,
                    mode()
                ),
                expected,
                "split {at}"
            );
        }
        let chars: Vec<String> = input.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(
            run(&chars, UnifiedParserStartingState::None, mode()),
            expected
        );
        // A prompt-opened thought: markers inside the payload stay payload.
        assert_eq!(
            run(
                &["Pick.</think>{\"city\":\"a </think> b\"}"],
                UnifiedParserStartingState::Reasoning,
                mode()
            ),
            vec![
                reasoning("Pick."),
                call("get_weather", json!({"city": "a </think> b"}))
            ]
        );
    }

    /// The reference parser maps a `null` value to JSON null before it looks at
    /// the schema, so a string-typed parameter gets null too.
    #[test]
    fn null_is_json_null_for_every_declared_type() {
        let input = "<tool_call><function=get_weather>\
             <parameter=city>NULL</parameter><parameter=days>null</parameter>\
             <parameter=note> null </parameter></function></tool_call>";
        assert_eq!(
            native(&[input]),
            vec![call(
                "get_weather",
                json!({"city": null, "days": null, "note": " null "})
            )]
        );
    }

    #[test]
    fn union_types_parse_as_json_and_untyped_schemas_stay_strings() {
        let tools = vec![Tool {
            name: "f".to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": {
                "n": {"type": ["integer", "null"]},
                "b": {"type": ["boolean", "null"]},
                "choice": {"anyOf": [{"type": "integer"}, {"type": "string"}]}
            }}),
            strict: None,
        }];
        let mut parser = mimo_unified(&tools);
        let input = "<tool_call><function=f><parameter=n>42</parameter>\
             <parameter=b>true</parameter><parameter=choice>7</parameter>\
             </function></tool_call>";
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(input, &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(
            assemble(&output.events),
            vec![call("f", json!({"n": 42, "b": true, "choice": "7"}))]
        );
    }

    #[test]
    fn a_value_spelling_function_close_stays_in_the_value() {
        let input = "<tool_call><function=edit_file><parameter=path>a.rs</parameter>\
             <parameter=body>before </function> after</parameter></function></tool_call>";
        let expected = vec![call(
            "edit_file",
            json!({"path": "a.rs", "body": "before </function> after"}),
        )];
        assert_eq!(native(&[input]), expected);
        for (at, _) in input.char_indices().skip(1) {
            assert_eq!(
                native(&[&input[..at], &input[at..]]),
                expected,
                "split {at}"
            );
        }
    }

    #[test]
    fn integers_beyond_i64_stay_numbers() {
        let tools = vec![Tool {
            name: "f".to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
            strict: None,
        }];
        let mut parser = mimo_unified(&tools);
        let mut output = UnifiedParserOutput::default();
        parser
            .parse_into(
                "<tool_call><function=f><parameter=n>9223372036854775808</parameter></function></tool_call>",
                &mut output,
            )
            .unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(
            assemble(&output.events),
            vec![call("f", json!({"n": 9223372036854775808u64}))]
        );
    }

    #[test]
    fn non_finite_numbers_stay_strings() {
        let input = "<tool_call><function=get_weather>\
             <parameter=ratio>NaN</parameter><parameter=days>1e400</parameter>\
             </function></tool_call>";
        let tools = vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": {
                "ratio": {"type": "number"},
                "days": {"type": "number"}
            }}),
            strict: None,
        }];
        let mut parser = mimo_unified(&tools);
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(input, &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(
            assemble(&output.events),
            vec![call(
                "get_weather",
                json!({"ratio": "NaN", "days": "1e400"})
            )]
        );
    }

    #[test]
    fn html5_references_decode_like_python_html_unescape() {
        let input = "<tool_call><function=edit_file>\
             <parameter=body>&#128; &lt x&ltx &#0;</parameter>\
             </function></tool_call>";
        assert_eq!(
            native(&[input]),
            vec![call(
                "edit_file",
                json!({"body": "\u{20ac} < x<x \u{fffd}"})
            )]
        );
    }

    #[test]
    fn html_entities_decode_like_the_reference_parser() {
        let input = "<tool_call><function=edit_file>\
             <parameter=body>  &#10;&#x41;&copy;&amp;lt; &hellip;&nosuch; &  \n</parameter>\
             </function></tool_call>";
        assert_eq!(
            native(&[input]),
            vec![call(
                "edit_file",
                json!({"body": "  \nA\u{a9}&lt; \u{2026}&nosuch; &  \n"})
            )]
        );
    }

    #[test]
    fn reset_returns_unconsumed_text() {
        let mut parser = mimo_unified(&tools());
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("abc <tool_c", &mut output).unwrap();
        assert_eq!(parser.reset(), "<tool_c");
        parser.parse_into("plain", &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(assemble(&output.events), vec![text("abc plain")]);
    }
}

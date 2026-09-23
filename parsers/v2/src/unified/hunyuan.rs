// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified parser for Tencent Hunyuan (Hy3) output.
//!
//! ```text
//! reasoning:  <think:opensource>…</think:opensource>
//! tool calls: <tool_calls:opensource>
//!             <tool_call:opensource>NAME<tool_sep:opensource>
//!             <arg_key:opensource>K</arg_key:opensource>
//!             <arg_value:opensource>V</arg_value:opensource>
//!             </tool_call:opensource>
//!             </tool_calls:opensource>
//! ```
//!
//! The Hy3 tokenizer spells every marker with the `:opensource` suffix; they are
//! ordinary added tokens, not special ones. Newer checkpoints drop
//! `<tool_sep>`, which is optional here.
//!
//! The scan core is the shared `WrappedBlockScanner`. The family adds an invoke
//! boundary that knows the argument slots: inside `<arg_value>` only its closing
//! marker is markup, so a value that spells `</tool_call>` (or any other marker)
//! reaches the tool byte for byte instead of ending the call. Values are typed
//! from the tool schema the way the engine's detector does, and a call naming a
//! tool the request did not offer is dropped.

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeBoundary, InvokeBoundaryFactory, InvokeEmitter, ReasoningSpec,
    WrappedBlockScanner, WrappedBlockSpec,
};
use crate::tool_calling::scan::{GuidedInvokePrefix, GuidedInvokePrefixContext};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::unified::{GuidedRouted, ScannerUnified, UnifiedParser};

const THINK_START: &str = "<think:opensource>";
const THINK_END: &str = "</think:opensource>";
const BLOCK_START: &str = "<tool_calls:opensource>";
const BLOCK_END: &str = "</tool_calls:opensource>";
const INVOKE_START: &str = "<tool_call:opensource>";
const INVOKE_END: &str = "</tool_call:opensource>";
const TOOL_SEP: &str = "<tool_sep:opensource>";
const KEY_START: &str = "<arg_key:opensource>";
const KEY_END: &str = "</arg_key:opensource>";
const VALUE_START: &str = "<arg_value:opensource>";
const VALUE_END: &str = "</arg_value:opensource>";

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

/// Finds the end of one invoke while skipping argument values, and classifies a
/// `<tool_call>` header that precedes guided JSON.
#[derive(Default)]
struct HunyuanInvocationBoundary {
    /// Inside an `<arg_value>`, whose body only its closing marker ends.
    in_value: bool,
    scan_from: usize,
    guided_prefix_scan_from: usize,
    /// First markup or JSON byte after a guided-mode header, kept once found so a
    /// later append cannot reclassify the header.
    guided_prefix_first: Option<(usize, bool)>,
}

impl InvokeBoundary for HunyuanInvocationBoundary {
    fn owns_guided_prefix(&self) -> bool {
        true
    }

    /// Under guided decoding the payload is bare JSON, so a `<tool_call>` header
    /// ahead of it is stray framing. Inside a thought, or ahead of a competing
    /// marker, only the header marker goes and the text after it stays. Outside a
    /// thought, a header whose JSON arrives before any markup is removed up to the
    /// payload; one followed by markup is a native call.
    fn guided_prefix_append(
        &mut self,
        candidate: &str,
        append: &str,
        context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        let header = candidate.strip_prefix(INVOKE_START)?;
        if !context.outside_reasoning || context.followed_by_competing_marker {
            return Some(GuidedInvokePrefix::Strip(INVOKE_START.len()));
        }
        if self.guided_prefix_first.is_none() {
            let append_start = candidate.len() - append.len();
            let scan_from = self
                .guided_prefix_scan_from
                .max(append_start.saturating_sub(INVOKE_START.len()));
            self.guided_prefix_scan_from = header.len();
            self.guided_prefix_first = header[scan_from..]
                .find(['{', '[', '<'])
                .map(|at| (scan_from + at, header.as_bytes()[scan_from + at] == b'<'));
        }
        match self.guided_prefix_first {
            Some((_, true)) => Some(GuidedInvokePrefix::NoMatch),
            Some((payload_at, false)) => Some(if context.payload_is_empty {
                GuidedInvokePrefix::Match(INVOKE_START.len() + payload_at)
            } else {
                GuidedInvokePrefix::Strip(INVOKE_START.len() + payload_at)
            }),
            None => Some(GuidedInvokePrefix::Pending),
        }
    }

    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        _flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        loop {
            if self.in_value {
                let Some(end) = find_from(candidate, self.scan_from, VALUE_END) else {
                    self.scan_from = next_scan_start(candidate, VALUE_END.len());
                    return None;
                };
                self.in_value = false;
                self.scan_from = end + VALUE_END.len();
                continue;
            }
            let close = find_from(candidate, self.scan_from, INVOKE_END);
            let value = find_from(candidate, self.scan_from, VALUE_START);
            match (close, value) {
                (Some(close), Some(value)) if close < value => {
                    return Some(close + INVOKE_END.len());
                }
                (_, Some(value)) => {
                    self.in_value = true;
                    self.scan_from = value + VALUE_START.len();
                }
                (Some(close), None) => return Some(close + INVOKE_END.len()),
                (None, None) => {
                    self.scan_from =
                        next_scan_start(candidate, INVOKE_END.len().max(VALUE_START.len()));
                    return None;
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
    Box::new(HunyuanInvocationBoundary::default())
}

/// The schema of one argument: a direct `properties` entry, else the first one
/// found through the object schema's top-level `allOf` / `anyOf` / `oneOf`.
fn argument_schema<'a>(
    parameters: &'a serde_json::Value,
    argument: &str,
) -> Option<&'a serde_json::Value> {
    if let Some(schema) = parameters
        .get("properties")
        .and_then(|properties| properties.get(argument))
    {
        return Some(schema);
    }
    ["allOf", "anyOf", "oneOf"]
        .iter()
        .filter_map(|keyword| parameters.get(*keyword)?.as_array())
        .flatten()
        .find_map(|branch| branch.get("properties")?.get(argument))
}

/// The JSON Schema types one argument may take, with `null` removed. A schema with
/// no `type` falls back to its `anyOf` / `oneOf` options, then to `string`.
fn declared_types(tools: &[Tool], function: &str, argument: &str) -> Vec<String> {
    let schema = tools
        .iter()
        .find(|tool| tool.name == function)
        .and_then(|tool| argument_schema(&tool.parameters, argument));
    let mut types = Vec::new();
    let mut collect = |schema: &serde_json::Value| match schema.get("type") {
        Some(serde_json::Value::String(name)) => types.push(name.clone()),
        Some(serde_json::Value::Array(names)) => types.extend(
            names
                .iter()
                .filter_map(|name| name.as_str().map(String::from)),
        ),
        _ => types.push("string".to_string()),
    };
    match schema {
        Some(schema) if schema.get("type").is_some() => collect(schema),
        Some(schema) => match schema.get("anyOf").or_else(|| schema.get("oneOf")) {
            Some(serde_json::Value::Array(options)) => options.iter().for_each(collect),
            _ => collect(schema),
        },
        None => collect(&serde_json::Value::Null),
    }
    types
        .into_iter()
        .map(|name| {
            let lower = name.to_ascii_lowercase();
            match lower.as_str() {
                "str" | "text" | "varchar" | "char" | "enum" => "string".to_string(),
                "bool" | "binary" => "boolean".to_string(),
                "list" => "array".to_string(),
                "dict" | "map" => "object".to_string(),
                "double" => "number".to_string(),
                _ if ["int", "uint", "long", "short", "unsigned"]
                    .iter()
                    .any(|prefix| lower.starts_with(prefix)) =>
                {
                    "integer".to_string()
                }
                _ if lower.starts_with("num") || lower.starts_with("float") => "number".to_string(),
                _ => name,
            }
        })
        .filter(|name| name != "null")
        .collect()
}

/// Type one raw argument value: boolean, integer, number, then JSON for compound
/// types, then the string itself. A string-typed value is the model's bytes verbatim.
fn typed_value(raw: &str, types: &[String]) -> serde_json::Value {
    let has = |name: &str| types.iter().any(|declared| declared == name);
    if has("boolean") {
        match raw.to_ascii_lowercase().as_str() {
            "true" => return serde_json::Value::Bool(true),
            "false" => return serde_json::Value::Bool(false),
            _ => {}
        }
    }
    if has("integer")
        && let Ok(number) = raw.trim().parse::<i64>()
    {
        return number.into();
    }
    if has("number") {
        let trimmed = raw.trim();
        let parsed = if trimmed.contains(['.', 'e', 'E']) {
            trimmed
                .parse::<f64>()
                .ok()
                .filter(|number| number.is_finite())
                .map(serde_json::Value::from)
        } else {
            trimmed.parse::<i64>().ok().map(serde_json::Value::from)
        };
        if let Some(number) = parsed {
            return number;
        }
    }
    let compound = types
        .iter()
        .any(|name| !matches!(name.as_str(), "string" | "boolean" | "integer" | "number"));
    if (compound || !has("string"))
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(raw)
        && json_type_is_declared(&value, types)
    {
        return value;
    }
    serde_json::Value::String(raw.to_string())
}

/// Whether a parsed JSON value has a declared type. `null` is always accepted
/// (nullable unions drop it from the list), a type this parser does not know
/// accepts anything, and an integer satisfies `number`.
fn json_type_is_declared(value: &serde_json::Value, types: &[String]) -> bool {
    let has = |name: &str| types.iter().any(|declared| declared == name);
    let known = ["string", "boolean", "integer", "number", "array", "object"];
    if types
        .iter()
        .any(|declared| !known.contains(&declared.as_str()))
    {
        return true;
    }
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Bool(_) => has("boolean"),
        serde_json::Value::Number(number) => {
            has("number") || (has("integer") && (number.is_i64() || number.is_u64()))
        }
        serde_json::Value::String(_) => has("string"),
        serde_json::Value::Array(_) => has("array"),
        serde_json::Value::Object(_) => has("object"),
    }
}

/// Split one complete invoke into its function name and raw `(key, value)`
/// pairs, or `None` when an argument slot is left open.
fn read_invoke(invoke: &str) -> Option<(&str, Vec<(&str, &str)>)> {
    let body = invoke
        .strip_prefix(INVOKE_START)?
        .strip_suffix(INVOKE_END)?;
    let name_end = [TOOL_SEP, KEY_START]
        .iter()
        .filter_map(|marker| body.find(marker))
        .min()
        .unwrap_or(body.len());
    let name = body[..name_end].trim();
    let mut rest = &body[name_end..];
    let mut arguments = Vec::new();
    while let Some(key_at) = rest.find(KEY_START) {
        let after_key = &rest[key_at + KEY_START.len()..];
        let key_len = after_key.find(KEY_END)?;
        let key = after_key[..key_len].trim();
        // The value opener follows its own key, layout whitespace aside; a key
        // with no value of its own leaves the invoke malformed rather than taking
        // the next argument's value.
        let value = after_key[key_len + KEY_END.len()..]
            .trim_start()
            .strip_prefix(VALUE_START)?;
        let value_len = value.find(VALUE_END)?;
        arguments.push((key, &value[..value_len]));
        rest = &value[value_len + VALUE_END.len()..];
    }
    Some((name, arguments))
}

/// Types each complete `<tool_call>` invoke the scan core delimits.
struct HunyuanEmitter {
    tools: Vec<Tool>,
}

impl InvokeEmitter for HunyuanEmitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let Some((name, raw_arguments)) = read_invoke(invoke) else {
            tracing::warn!(
                why = "hunyuan_unparsable_invoke",
                "Hunyuan tool-call invoke decoded to no call"
            );
            return Ok(None);
        };
        let offered = self.tools.is_empty() || self.tools.iter().any(|tool| tool.name == name);
        if name.is_empty() || !offered {
            tracing::warn!(name, "Hunyuan tool call names no offered tool");
            return Ok(None);
        }
        let mut arguments = serde_json::Map::new();
        for (key, raw) in raw_arguments {
            let types = declared_types(&self.tools, name, key);
            arguments.insert(key.to_string(), typed_value(raw, &types));
        }
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(name.to_string()),
            arguments: serde_json::Value::Object(arguments).to_string(),
            complete: true,
        }))
    }
}

/// Build the Hunyuan unified parser for one stream.
pub(crate) fn hunyuan_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    let spec = WrappedBlockSpec {
        family: "hunyuan",
        block_starts: vec![BLOCK_START.into()],
        block_ends: vec![BLOCK_END.into()],
        invoke_start: INVOKE_START.into(),
        invoke_end: INVOKE_END.into(),
        // Control markers outside a block are stray markup and never shown.
        orphan_markers: vec![
            BLOCK_END.into(),
            INVOKE_END.into(),
            TOOL_SEP.into(),
            KEY_START.into(),
            KEY_END.into(),
            VALUE_START.into(),
            VALUE_END.into(),
        ],
        holdback_markers: vec![
            BLOCK_START.into(),
            BLOCK_END.into(),
            INVOKE_START.into(),
            INVOKE_END.into(),
            TOOL_SEP.into(),
            KEY_START.into(),
            KEY_END.into(),
            VALUE_START.into(),
            VALUE_END.into(),
        ],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(invocation_boundary)),
        preserve_special_tokens: false,
        ..Default::default()
    };
    let scanner = WrappedBlockScanner::new(
        spec,
        HunyuanEmitter {
            tools: tools.to_vec(),
        },
    )
    .with_reasoning(ReasoningSpec {
        start: THINK_START,
        end: THINK_END,
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

    const SFX: &str = ":opensource";

    fn tools() -> Vec<Tool> {
        let tool = |name: &str, properties: serde_json::Value| Tool {
            name: name.to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": properties}),
            strict: None,
        };
        vec![
            tool(
                "get_weather",
                json!({
                    "city": {"type": "string"},
                    "date": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                    "tags": {"type": "array"}
                }),
            ),
            tool("get_current_date", json!({})),
            tool("search", json!({"query": {"type": "string"}})),
        ]
    }

    /// Spell a bare-marker fixture the way the shipping tokenizer does.
    fn suffixed(bare: &str) -> String {
        let mut out = bare.to_string();
        for name in [
            "think",
            "tool_calls",
            "tool_call",
            "tool_sep",
            "arg_key",
            "arg_value",
        ] {
            out = out
                .replace(&format!("<{name}>"), &format!("<{name}{SFX}>"))
                .replace(&format!("</{name}>"), &format!("</{name}{SFX}>"));
        }
        out
    }

    fn run(
        chunks: &[&str],
        starting_state: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
    ) -> Vec<UnifiedEvent> {
        let mut parser = hunyuan_unified(&tools());
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

    /// Every two-chunk split of `input`, plus one character at a time, must
    /// assemble to the same events as the whole input.
    fn assert_split_invariant(input: &str, starting_state: UnifiedParserStartingState) {
        let whole = run(&[input], starting_state, UnifiedToolOutputMode::Native);
        for (at, _) in input.char_indices().skip(1) {
            let got = run(
                &[&input[..at], &input[at..]],
                starting_state,
                UnifiedToolOutputMode::Native,
            );
            assert_eq!(got, whole, "split at byte {at} of {input:?}");
        }
        let chars: Vec<String> = input.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(
            run(&chars, starting_state, UnifiedToolOutputMode::Native),
            whole,
            "char-at-a-time {input:?}"
        );
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
        let input = suffixed("<think>Let me think.</think>The answer is 42.");
        assert_eq!(
            native(&[&input]),
            vec![reasoning("Let me think."), text("The answer is 42.")]
        );
    }

    #[test]
    fn prompt_opened_reasoning_closes_without_an_opener() {
        let input = suffixed("Let me think.</think>The answer is 42.");
        assert_eq!(
            run(
                &[&input],
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::Native
            ),
            vec![reasoning("Let me think."), text("The answer is 42.")]
        );
    }

    #[test]
    fn zero_argument_call_inline_and_with_newlines() {
        for bare in [
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
            "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>",
        ] {
            let input = suffixed(bare);
            assert_eq!(
                native(&[&input]),
                vec![call("get_current_date", json!({}))],
                "{input:?}"
            );
        }
    }

    #[test]
    fn single_call_with_template_layout() {
        let input = suffixed(
            "<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n\
             <arg_value>Beijing</arg_value>\n<arg_key>date</arg_key>\n\
             <arg_value>2026-03-30</arg_value>\n</tool_call>\n</tool_calls>",
        );
        assert_eq!(
            native(&[&input]),
            vec![call(
                "get_weather",
                json!({"city": "Beijing", "date": "2026-03-30"})
            )]
        );
    }

    #[test]
    fn call_without_tool_separator() {
        let input = suffixed(
            "<tool_calls><tool_call>get_weather\n<arg_key>city</arg_key>\n\
             <arg_value>Beijing</arg_value>\n</tool_call></tool_calls>",
        );
        assert_eq!(
            native(&[&input]),
            vec![call("get_weather", json!({"city": "Beijing"}))]
        );
    }

    #[test]
    fn content_before_tool_call_is_kept() {
        let input = suffixed(
            "Checking.<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>",
        );
        assert_eq!(
            native(&[&input]),
            vec![text("Checking."), call("get_current_date", json!({}))]
        );
    }

    #[test]
    fn multiple_tool_calls_keep_order() {
        let input = suffixed(
            "<tool_calls>\
             <tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>\n\
             <tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Hangzhou</arg_value></tool_call>\n\
             <tool_call>get_current_date<tool_sep></tool_call>\
             </tool_calls>",
        );
        assert_eq!(
            native(&[&input]),
            vec![
                call("get_weather", json!({"city": "Beijing"})),
                call("get_weather", json!({"city": "Hangzhou"})),
                call("get_current_date", json!({})),
            ]
        );
    }

    #[test]
    fn arguments_are_typed_from_the_schema() {
        let input = suffixed(
            "<tool_calls><tool_call>get_weather<tool_sep>\
             <arg_key>city</arg_key><arg_value>123</arg_value>\
             <arg_key>days</arg_key><arg_value>3</arg_value>\
             <arg_key>metric</arg_key><arg_value>true</arg_value>\
             <arg_key>tags</arg_key><arg_value>[\"a\", \"b\"]</arg_value>\
             </tool_call></tool_calls>",
        );
        assert_eq!(
            native(&[&input]),
            vec![call(
                "get_weather",
                json!({"city": "123", "days": 3, "metric": true, "tags": ["a", "b"]})
            )]
        );
    }

    #[test]
    fn reasoning_then_tool_call() {
        let input = suffixed(
            "<think>Need the date.</think>\n<tool_calls>\n<tool_call>get_current_date<tool_sep>\n\
             </tool_call>\n</tool_calls>",
        );
        let events = native(&[&input]);
        assert_eq!(events[0], reasoning("Need the date."));
        assert_eq!(events.last().unwrap(), &call("get_current_date", json!({})));
    }

    #[test]
    fn tool_call_before_think_close_ends_the_thought() {
        let input = suffixed(
            "Need the date.<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
        );
        assert_eq!(
            run(
                &[&input],
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::Native
            ),
            vec![
                reasoning("Need the date."),
                call("get_current_date", json!({}))
            ]
        );
    }

    #[test]
    fn think_marker_inside_an_argument_value_is_value_text() {
        let input = suffixed(
            "<tool_calls><tool_call>search<tool_sep><arg_key>query</arg_key>\
             <arg_value>what does <think> mean</arg_value></tool_call></tool_calls>",
        );
        let expected = format!("what does <think{SFX}> mean");
        assert_eq!(
            native(&[&input]),
            vec![call("search", json!({"query": expected}))]
        );
    }

    #[test]
    fn every_marker_inside_an_argument_value_is_value_text() {
        for literal in [
            "<tool_sep>",
            "<arg_key>k</arg_key>",
            "<arg_value>",
            "<tool_call>inner</tool_call>",
            "</tool_calls>",
            "a &quot;quoted&quot; &lt;tag&gt; &amp; more",
        ] {
            for literal in [literal.to_string(), suffixed(literal)] {
                let value = format!("  before {literal} after\n");
                let input = format!(
                    "{}{value}{}",
                    suffixed(
                        "<tool_calls><tool_call>search<tool_sep><arg_key>query</arg_key><arg_value>"
                    ),
                    suffixed("</arg_value></tool_call></tool_calls>")
                );
                assert_eq!(
                    native(&[&input]),
                    vec![call("search", json!({"query": value}))],
                    "{literal}"
                );
                assert_split_invariant(&input, UnifiedParserStartingState::None);
            }
        }
    }

    #[test]
    fn a_key_without_its_own_value_is_malformed() {
        let input = suffixed(
            "<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key>\
             <arg_key>days</arg_key><arg_value>3</arg_value></tool_call></tool_calls>",
        );
        assert_eq!(native(&[&input]), vec![]);
    }

    #[test]
    fn composed_parameter_schemas_type_arguments() {
        let tools = vec![Tool {
            name: "f".to_string(),
            description: None,
            parameters: json!({"allOf": [
                {"type": "object", "properties": {"x": {"type": "integer"}}}
            ]}),
            strict: None,
        }];
        let input = suffixed(
            "<tool_calls><tool_call>f<tool_sep><arg_key>x</arg_key>\
             <arg_value>3</arg_value></tool_call></tool_calls>",
        );
        let mut parser = hunyuan_unified(&tools);
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(&input, &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(assemble(&output.events), vec![call("f", json!({"x": 3}))]);
    }

    #[test]
    fn stray_control_markers_outside_a_block_are_not_shown() {
        for marker in [
            "</tool_call>",
            "<tool_sep>",
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
        ] {
            let input = format!("before{}after", suffixed(marker));
            assert_eq!(native(&[&input]), vec![text("beforeafter")], "{marker}");
            assert_split_invariant(&input, UnifiedParserStartingState::None);
        }
    }

    #[test]
    fn guided_mode_never_reads_native_argument_json_as_a_call() {
        let input = suffixed(
            r#"<tool_calls><tool_call>search<tool_sep><arg_key>query</arg_key><arg_value>{"name":"get_weather","arguments":{"city":"Paris"}}</arg_value></tool_call></tool_calls>"#,
        );
        let mode = || UnifiedToolOutputMode::GuidedJson { named_tool: None };
        let whole = run(&[&input], UnifiedParserStartingState::None, mode());
        assert!(
            whole.iter().all(|event| !matches!(
                event,
                UnifiedEvent::ToolCall { name, .. } if name == "get_weather"
            )),
            "{whole:?}"
        );
        for (at, _) in input.char_indices().skip(1) {
            assert_eq!(
                run(
                    &[&input[..at], &input[at..]],
                    UnifiedParserStartingState::None,
                    mode()
                ),
                whole,
                "split {at}"
            );
        }
    }

    #[test]
    fn values_of_an_undeclared_type_stay_strings() {
        let tools = vec![Tool {
            name: "f".to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": {
                "n": {"type": "integer"},
                "xs": {"type": "array"},
                "r": {"type": "number"}
            }}),
            strict: None,
        }];
        let input = suffixed(
            "<tool_calls><tool_call>f<tool_sep>\
             <arg_key>n</arg_key><arg_value>true</arg_value>\
             <arg_key>xs</arg_key><arg_value>3</arg_value>\
             <arg_key>r</arg_key><arg_value>1e309</arg_value>\
             </tool_call></tool_calls>",
        );
        let mut parser = hunyuan_unified(&tools);
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(&input, &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(
            assemble(&output.events),
            vec![call("f", json!({"n": "true", "xs": "3", "r": "1e309"}))]
        );
    }

    #[test]
    fn text_after_the_calls_is_kept() {
        let input = suffixed(
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>Done.",
        );
        assert_eq!(
            native(&[&input]),
            vec![call("get_current_date", json!({})), text("Done.")]
        );
    }

    #[test]
    fn chunk_boundaries_never_change_the_result() {
        let calls = suffixed(
            "<think>Compare two cities.</think>I'll check both.<tool_calls>\n\
             <tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n\
             <arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>\n\
             <tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Hangzhou</arg_value>\n\
             </tool_call>\n</tool_calls>",
        );
        assert_split_invariant(&calls, UnifiedParserStartingState::None);
        let whole = native(&[&calls]);
        assert_eq!(
            whole,
            vec![
                reasoning("Compare two cities."),
                text("I'll check both."),
                call("get_weather", json!({"city": "Beijing", "days": 3})),
                call("get_weather", json!({"city": "Hangzhou"})),
            ]
        );

        let answer = suffixed("Short thought.</think>Use a < b, then <b>bold</b>.");
        assert_split_invariant(&answer, UnifiedParserStartingState::Reasoning);
    }

    #[test]
    fn split_marker_is_held_until_decided() {
        let mut parser = hunyuan_unified(&tools());
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("Hello <tool_ca", &mut output).unwrap();
        assert_eq!(assemble(&output.events), vec![text("Hello ")]);
        parser.parse_into("ke is good", &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(
            assemble(&output.events),
            vec![text("Hello <tool_cake is good")]
        );
    }

    #[test]
    fn unterminated_reasoning_is_reasoning() {
        let input = suffixed("<think>still thinking when the budget ran out");
        assert_eq!(
            native(&[&input]),
            vec![reasoning("still thinking when the budget ran out")]
        );
    }

    #[test]
    fn unterminated_tool_call_is_dropped_without_leaking_markup() {
        let input = suffixed(
            "On it.<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>\n<arg_value>Bei",
        );
        assert_eq!(native(&[&input]), vec![text("On it.")]);
    }

    #[test]
    fn partial_marker_at_end_of_stream_is_text() {
        assert_eq!(native(&["value <tool_cal"]), vec![text("value <tool_cal")]);
    }

    #[test]
    fn unknown_tool_is_not_a_call() {
        let input = suffixed(
            "<tool_calls><tool_call>nonexistent<tool_sep></tool_call>\
             <tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
        );
        let events = native(&[&input]);
        let calls: Vec<_> = events
            .iter()
            .filter(|event| matches!(event, UnifiedEvent::ToolCall { .. }))
            .collect();
        assert_eq!(calls, vec![&call("get_current_date", json!({}))]);
    }

    #[test]
    fn guided_named_choice_reads_bare_arguments_after_reasoning() {
        let input = suffixed("Pick Paris.</think>{\"city\": \"Paris\"}");
        assert_eq!(
            run(
                &[&input[..20], &input[20..]],
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
    }

    #[test]
    fn guided_required_choice_reads_call_array_and_recovers_malformed_as_text() {
        let mode = UnifiedToolOutputMode::GuidedJson { named_tool: None };
        assert_eq!(
            run(
                &[
                    r#"[{"name":"search","arguments":{"query":"a"}},"#,
                    r#"{"name":"get_current_date","parameters":{}}]"#
                ],
                UnifiedParserStartingState::None,
                mode.clone()
            ),
            vec![
                call("search", json!({"query": "a"})),
                call("get_current_date", json!({}))
            ]
        );
        assert_eq!(
            run(
                &[r#"[{"name":"search","arguments":"#],
                UnifiedParserStartingState::None,
                mode
            ),
            vec![text(r#"[{"name":"search","arguments":"#)]
        );
    }

    #[test]
    fn guided_payload_keeps_reasoning_markers_inside_json_strings() {
        let named = |payload: &str, start| {
            run(
                &[payload],
                start,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("search".into()),
                },
            )
        };
        for marker in [
            "</think>",
            "<think>",
            "</think:opensource>",
            "<think:opensource>",
        ] {
            let query = format!("literal {marker} text");
            let payload = json!({"query": query}).to_string();
            assert_eq!(
                named(&payload, UnifiedParserStartingState::None),
                vec![call("search", json!({"query": query}))],
                "{marker}"
            );
            let after_thought = format!("Thinking.</think:opensource>{payload}");
            assert_eq!(
                named(&after_thought, UnifiedParserStartingState::Reasoning),
                vec![
                    reasoning("Thinking."),
                    call("search", json!({"query": query}))
                ],
                "{marker} after a thought"
            );
        }
        // A generated thought ahead of the payload is still reasoning.
        let payload = json!({"query": "a </think> b"}).to_string();
        let input = format!("\n<think:opensource>Hm.</think:opensource>\n{payload}");
        for at in 1..input.len() {
            if !input.is_char_boundary(at) {
                continue;
            }
            assert_eq!(
                run(
                    &[&input[..at], &input[at..]],
                    UnifiedParserStartingState::None,
                    UnifiedToolOutputMode::GuidedJson {
                        named_tool: Some("search".into()),
                    },
                ),
                vec![
                    reasoning("Hm."),
                    call("search", json!({"query": "a </think> b"}))
                ],
                "split {at}"
            );
        }
    }

    #[test]
    fn reset_returns_unconsumed_text() {
        let mut parser = hunyuan_unified(&tools());
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("abc <tool_c", &mut output).unwrap();
        assert_eq!(parser.reset(), "<tool_c");
        parser.parse_into("plain", &mut output).unwrap();
        let mut tail = parser.finish().unwrap();
        output.append(&mut tail);
        assert_eq!(assemble(&output.events), vec![text("abc plain")]);
    }
}

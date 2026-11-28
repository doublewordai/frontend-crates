// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GLM-4.7 Tool Call Parser
// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
// Reference: https://huggingface.co/zai-org/GLM-4.7/blob/main/chat_template.jinja

use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use tracing::warn;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::Glm47ParserConfig;
use super::OrderedArguments;
use super::parsed_value::{ParsedValue, coerce_integer_literal, is_integer_literal};
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Render a tool_call block snippet for logs. Bounded so a huge truncated
/// argument body doesn't blow up the log line; control chars are escaped
/// because raw newlines/tabs make the warning unreadable in grep/jq.
fn truncate_for_log(s: &str) -> String {
    const MAX: usize = 200;
    let mut out = String::with_capacity(MAX.min(s.len()) + 16);
    let mut bytes = 0usize;
    for ch in s.chars() {
        if bytes >= MAX {
            out.push('…');
            break;
        }
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
        bytes += ch.len_utf8();
    }
    out
}

/// Check if a chunk contains the start of a GLM-4.7 tool call.
/// Format: <tool_call>function_name<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>
pub fn detect_tool_call_start_glm47(chunk: &str, config: &Glm47ParserConfig) -> bool {
    let start_token = &config.tool_call_start;
    let arg_key_start = &config.arg_key_start;

    // Check if we have the complete start token
    if chunk.contains(start_token.as_str()) || chunk.contains(arg_key_start.as_str()) {
        return true;
    }

    // Check for partial match at the end of the chunk (for streaming)
    for i in 1..start_token.len() {
        if chunk.ends_with(&start_token[..i]) {
            return true;
        }
    }

    false
}

/// Find the end position of all consecutive GLM-4.7 tool calls.
/// When a model emits multiple parallel tool calls in one chunk
/// (e.g. `<tool_call>A</tool_call><tool_call>B</tool_call>`), this
/// function advances past every consecutive start→end pair so the
/// entire group is captured as a single jailed region.  Returns the
/// position after the last `</tool_call>` found, or the length of the
/// chunk when no end token is present.
pub fn find_tool_call_end_position_glm47(chunk: &str, config: &Glm47ParserConfig) -> usize {
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    if !chunk.contains(start_token.as_str()) && chunk.contains(config.arg_key_start.as_str()) {
        return find_bare_glm47_tool_call_end_position(chunk, config).unwrap_or(chunk.len());
    }

    let Some(first_end) = chunk.find(end_token.as_str()) else {
        return chunk.len();
    };

    let mut cursor = first_end + end_token.len();

    loop {
        let rest = &chunk[cursor..];
        let trimmed = rest.trim_start();
        if !trimmed.starts_with(start_token.as_str()) {
            break;
        }
        let trim_offset = rest.len() - trimmed.len();
        let search_from = cursor + trim_offset + start_token.len();
        if let Some(end_pos) = chunk[search_from..].find(end_token.as_str()) {
            cursor = search_from + end_pos + end_token.len();
        } else {
            break;
        }
    }

    cursor
}

fn find_bare_glm47_tool_call_end_position(text: &str, config: &Glm47ParserConfig) -> Option<usize> {
    let marker_idx = first_orphan_glm47_marker_index(text, config)?;
    let before_marker = text[..marker_idx].trim_end();
    let function_name_start = before_marker
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);

    let candidate_name = before_marker[function_name_start..].trim();
    if candidate_name.is_empty()
        || !candidate_name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return None;
    }

    let mut cursor = function_name_start;
    let mut last_complete_end = None;
    while cursor < text.len() {
        let rest = &text[cursor..];
        let trim_offset = rest.len() - rest.trim_start().len();
        let call_start = cursor + trim_offset;
        let tail = &text[call_start..];
        let Some(end_pos) = tail.find(config.tool_call_end.as_str()) else {
            break;
        };

        let call_end = call_start + end_pos + config.tool_call_end.len();
        cursor = consume_glm47_close_markers(text, call_end, config);
        last_complete_end = Some(cursor);

        if first_orphan_glm47_marker_index(&text[cursor..], config).is_none() {
            break;
        }
    }

    last_complete_end
}

/// Try to parse GLM-4.7 formatted tool calls from a message.
/// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_glm47(
    message: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let (normal_text, tool_calls) = extract_tool_calls(message, config, tools)?;

    let normal_content = if normal_text.is_empty() {
        Some("".to_string())
    } else {
        Some(normal_text)
    };

    Ok((tool_calls, normal_content))
}

/// Extract tool calls and normal text from message.
fn extract_tool_calls(
    text: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_parts = Vec::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    if !text.contains(start_token.as_str())
        && let Some(marker_idx) = first_orphan_glm47_marker_index(text, config)
    {
        if let Some((prefix, mut parsed_calls)) =
            recover_bare_glm47_calls(text, marker_idx, config, tools)?
        {
            let recovered_calls = parsed_calls.len();
            warn!(
                why = "bare_body_recovery",
                recovered_calls,
                recovered_bytes = text.len() - prefix.len(),
                kept_prefix_bytes = prefix.len(),
                "GLM-4.7 parser recovered complete bare call body/bodies without <tool_call> start"
            );
            calls.append(&mut parsed_calls);
            return Ok((prefix, calls));
        }
        warn!(
            why = "GLM-4.7 tool-call marker found without <tool_call> start; dropping orphan marker tail so wire tags do not leak into normal_text",
            dropped_block = %truncate_for_log(&text[marker_idx..]),
            "GLM-4.7 parser dropping orphan tool-call marker tail"
        );
        return Ok((orphan_glm47_prefix(text, marker_idx), calls));
    }

    while cursor < text.len() {
        // Find next tool call start
        if let Some(start_pos) = text[cursor..].find(start_token.as_str()) {
            let abs_start = cursor + start_pos;
            let gap = &text[cursor..abs_start];
            if let Some((prefix, mut parsed_calls)) =
                recover_bare_glm47_calls_in_span(gap, config, tools)?
            {
                if calls.is_empty() {
                    normal_parts.push(prefix);
                }
                calls.append(&mut parsed_calls);
            } else if calls.is_empty() {
                if let Some(marker_idx) = first_orphan_glm47_marker_index(gap, config) {
                    normal_parts.push(orphan_glm47_prefix(gap, marker_idx));
                } else {
                    normal_parts.push(gap.to_string());
                }
            }

            // Only surface normal text that precedes the first parsed call.
            // Text after any </tool_call> is not response content; matches the
            // convention ported into the generic XML parser by PR #9350 and
            // vLLM's glm47_moe_tool_parser.

            // Find the corresponding end token
            if let Some(end_pos) = text[abs_start..].find(end_token.as_str()) {
                let abs_end = abs_start + end_pos + end_token.len();
                let block = &text[abs_start..abs_end];

                // Parse this tool call block. Unparseable blocks (malformed
                // <tool_call>...</tool_call> markup the parser can't extract)
                // are dropped — emitting the raw markup as normal_text leaks
                // wire tags downstream. vLLM and SGLang both drop on this
                // path; aligning Dynamo to that contract.
                match parse_tool_call_block(block, config, tools) {
                    Ok(parsed_call) => calls.push(parsed_call),
                    Err(e) => {
                        warn!(
                            reason = %e,
                            why = "block has open + close fence but content failed to parse \
                                   as a GLM-4.7 tool call (e.g. empty function name, \
                                   missing <arg_key>, malformed args); dropping to avoid \
                                   leaking wire tags through normal_text",
                            dropped_block = %truncate_for_log(block),
                            "GLM-4.7 parser dropping unparseable tool_call block"
                        );
                    }
                }

                cursor = abs_end;
            } else {
                // Recovery: outer </tool_call> absent (max_tokens / EOS
                // truncation). Gated on `allow_eof_recovery` so streaming
                // early-exit doesn't fire mid-stream. Also requires an
                // `<arg_key>` opener in the trailing slice as the structural
                // signal that a real tool call was emitted.
                let block = &text[abs_start..];
                let arg_key_start = &config.arg_key_start;
                if config.allow_eof_recovery && block.contains(arg_key_start.as_str()) {
                    match parse_tool_call_block(block, config, tools) {
                        Ok(parsed_call) => {
                            calls.push(parsed_call);
                            cursor = text.len();
                            continue;
                        }
                        Err(e) => {
                            warn!(
                                reason = %e,
                                why = "EOF recovery enabled and <arg_key> opener present, \
                                       but parse_tool_call_block failed on the truncated \
                                       tail; dropping to avoid leaking wire tags through \
                                       normal_text",
                                dropped_block = %truncate_for_log(block),
                                "GLM-4.7 parser dropping truncated tool_call block (recovery attempt failed)"
                            );
                        }
                    }
                } else {
                    // Either recovery disabled (production default for GLM-4.7)
                    // or no <arg_key> in the tail (so this is plausibly not a
                    // real tool call at all, just a stray <tool_call> token).
                    let reason = if !config.allow_eof_recovery {
                        "allow_eof_recovery=false (production default for GLM-4.7 to match \
                         vLLM/SGLang on truncated tool calls)"
                    } else {
                        "no <arg_key> in the tail after the <tool_call> start fence, so the \
                         block does not look like a structurally-real GLM-4.7 tool call"
                    };
                    warn!(
                        why = %reason,
                        dropped_block = %truncate_for_log(block),
                        "GLM-4.7 parser dropping truncated tool_call block (no end fence)"
                    );
                }
                // Drop the truncated/unrecoverable tail. Emitting the raw
                // <tool_call>...<arg_key>...<arg_value>... prefix as
                // normal_text would leak wire tags into message.content; vLLM
                // strips the same way on truncation.
                break;
            }
        } else {
            // No more tool calls
            let gap = &text[cursor..];
            if let Some((prefix, mut parsed_calls)) =
                recover_bare_glm47_calls_in_span(gap, config, tools)?
            {
                if calls.is_empty() {
                    normal_parts.push(prefix);
                }
                calls.append(&mut parsed_calls);
            } else if calls.is_empty() {
                if let Some(marker_idx) = first_orphan_glm47_marker_index(gap, config) {
                    normal_parts.push(orphan_glm47_prefix(gap, marker_idx));
                } else {
                    normal_parts.push(gap.to_string());
                }
            }
            break;
        }
    }

    let normal_text = normal_parts.join("");
    let normal_text = if calls.is_empty() {
        normal_text.trim().to_string()
    } else {
        normal_text
    };
    Ok((normal_text, calls))
}

fn recover_bare_glm47_calls_in_span(
    span: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    let Some(marker_idx) = first_orphan_glm47_marker_index(span, config) else {
        return Ok(None);
    };
    let recovered = recover_bare_glm47_calls(span, marker_idx, config, tools)?;
    if let Some((prefix, parsed_calls)) = recovered {
        let recovered_calls = parsed_calls.len();
        warn!(
            why = "bare_body_gap_recovery",
            recovered_calls,
            recovered_bytes = span.len() - prefix.len(),
            kept_prefix_bytes = prefix.len(),
            "GLM-4.7 parser recovered complete bare call body/bodies before a later <tool_call>"
        );
        return Ok(Some((prefix, parsed_calls)));
    }
    Ok(None)
}

fn first_orphan_glm47_marker_index(text: &str, config: &Glm47ParserConfig) -> Option<usize> {
    [
        config.tool_call_end.as_str(),
        config.arg_key_start.as_str(),
        config.arg_key_end.as_str(),
        config.arg_value_start.as_str(),
        config.arg_value_end.as_str(),
    ]
    .into_iter()
    .filter_map(|marker| text.find(marker))
    .min()
}

fn orphan_glm47_prefix(text: &str, marker_idx: usize) -> String {
    let prefix = text[..marker_idx].trim_end();
    let token_start = prefix
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);
    let tail = &prefix[token_start..];
    if !tail.is_empty()
        && tail
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        prefix[..token_start].trim().to_string()
    } else {
        prefix.trim().to_string()
    }
}

fn recover_bare_glm47_calls(
    text: &str,
    marker_idx: usize,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    if !text[marker_idx..].contains(config.tool_call_end.as_str()) {
        return Ok(None);
    }

    let before_marker = text[..marker_idx].trim_end();
    let function_name_start = before_marker
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);

    let candidate_name = before_marker[function_name_start..].trim();
    if candidate_name.is_empty()
        || !candidate_name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return Ok(None);
    }

    let prefix = text[..function_name_start].to_string();
    let mut cursor = function_name_start;
    let mut calls = Vec::new();

    while cursor < text.len() {
        let rest = &text[cursor..];
        let trim_offset = rest.len() - rest.trim_start().len();
        cursor += trim_offset;

        let tail = &text[cursor..];
        let Some(end_pos) = tail.find(config.tool_call_end.as_str()) else {
            break;
        };
        let call_end = cursor + end_pos + config.tool_call_end.len();
        let wrapped = format!("{}{}", config.tool_call_start, &text[cursor..call_end]);
        calls.push(parse_tool_call_block(&wrapped, config, tools)?);
        cursor = call_end;

        if is_glm47_close_marker_spam(&text[cursor..], config) {
            warn!(
                why = "orphan_close_marker_spam",
                dropped_block = %truncate_for_log(&text[cursor..]),
                "GLM-4.7 parser dropping orphan close-marker spam after recovered bare call"
            );
            break;
        }

        if first_orphan_glm47_marker_index(&text[cursor..], config).is_none() {
            break;
        }
    }

    if calls.is_empty() {
        return Ok(None);
    }
    Ok(Some((prefix, calls)))
}

fn is_glm47_close_marker_spam(text: &str, config: &Glm47ParserConfig) -> bool {
    let mut rest = text.trim_start();
    let mut saw_close = false;
    while let Some(after_close) = rest.strip_prefix(config.tool_call_end.as_str()) {
        saw_close = true;
        rest = after_close.trim_start();
    }
    saw_close && rest.is_empty()
}

fn consume_glm47_close_markers(text: &str, mut cursor: usize, config: &Glm47ParserConfig) -> usize {
    loop {
        let rest = &text[cursor..];
        let trim_offset = rest.len() - rest.trim_start().len();
        let close_start = cursor + trim_offset;
        if !text[close_start..].starts_with(config.tool_call_end.as_str()) {
            return cursor;
        }
        cursor = close_start + config.tool_call_end.len();
    }
}

/// Decode XML character entities in a string.
/// Handles the five predefined XML entities: &lt; &gt; &amp; &quot; &apos;
fn decode_xml_entities(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Coerce a raw string value using the tool's parameter schema.
/// Falls back to string if no schema is available or the type is unrecognized.
fn coerce_value(raw: &str, schema_type: Option<&str>) -> ParsedValue {
    let trimmed = raw.trim();

    // A `string` parameter is delivered verbatim: the model's text may
    // legitimately look like JSON (an object, array, or quoted text), and
    // parsing it would change its type behind the schema's back.
    if matches!(schema_type, Some("string")) {
        return Value::String(raw.to_string()).into();
    }

    // If the value already looks like JSON (object, array, or quoted string), parse it directly
    if (trimmed.starts_with('{') || trimmed.starts_with('[') || trimmed.starts_with('"'))
        && let Ok(v) = serde_json::from_str::<Value>(trimmed)
    {
        return v.into();
    }

    // Use schema type hints for coercion when available
    match schema_type {
        Some("integer") | Some("int") => {
            if let Some(value) = coerce_integer_literal(trimmed) {
                return value;
            }
        }
        Some("number") | Some("float") | Some("double") => {
            if let Some(value) = coerce_integer_literal(trimmed) {
                return value;
            }
            if let Ok(n) = trimmed.parse::<f64>()
                && let Some(num) = serde_json::Number::from_f64(n)
            {
                return Value::Number(num).into();
            }
        }
        Some("boolean") | Some("bool") => match trimmed.to_lowercase().as_str() {
            "true" | "1" | "yes" => return Value::Bool(true).into(),
            "false" | "0" | "no" => return Value::Bool(false).into(),
            _ => {}
        },
        Some("array") => {
            // Try JSON parse first, then fall back to comma-separated splitting
            if let Ok(v) = serde_json::from_str::<Value>(trimmed)
                && v.is_array()
            {
                return v.into();
            }
            let items: Vec<Value> = trimmed
                .split(',')
                .map(|s| Value::String(s.trim().to_string()))
                .collect();
            return Value::Array(items).into();
        }
        Some("null") if trimmed == "null" || trimmed == "None" || trimmed.is_empty() => {
            return Value::Null.into();
        }
        _ => {}
    }

    Value::String(raw.to_string()).into()
}

/// The JSON Schema types a parameter admits, as a bit set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SchemaTypes(u8);

impl SchemaTypes {
    const NULL: u8 = 1;
    const BOOLEAN: u8 = 1 << 1;
    const INTEGER: u8 = 1 << 2;
    const NUMBER: u8 = 1 << 3;
    const STRING: u8 = 1 << 4;
    const ARRAY: u8 = 1 << 5;
    const OBJECT: u8 = 1 << 6;

    /// The types a `type` name admits (with the aliases `coerce_value` accepts). A number
    /// admits integers.
    fn named(name: &str) -> Option<u8> {
        Some(match name {
            "null" => Self::NULL,
            "boolean" | "bool" => Self::BOOLEAN,
            "integer" | "int" => Self::INTEGER,
            "number" | "float" | "double" => Self::NUMBER | Self::INTEGER,
            "string" => Self::STRING,
            "array" => Self::ARRAY,
            "object" => Self::OBJECT,
            _ => return None,
        })
    }

    /// The types of an `enum` or `const` value. An integral number is also an integer.
    fn of_value(value: &Value) -> u8 {
        match value {
            Value::Null => Self::NULL,
            Value::Bool(_) => Self::BOOLEAN,
            Value::Number(n) if n.as_f64().is_some_and(|f| f.fract() != 0.0) => Self::NUMBER,
            Value::Number(_) => Self::NUMBER | Self::INTEGER,
            Value::String(_) => Self::STRING,
            Value::Array(_) => Self::ARRAY,
            Value::Object(_) => Self::OBJECT,
        }
    }

    fn admits(self, types: u8) -> bool {
        self.0 & types != 0
    }

    /// The `coerce_value` type name when the set holds a single type.
    fn single(self) -> Option<&'static str> {
        Some(match self.0 {
            Self::NULL => "null",
            Self::BOOLEAN => "boolean",
            Self::INTEGER => "integer",
            Self::NUMBER => "number",
            t if t == Self::NUMBER | Self::INTEGER => "number",
            Self::STRING => "string",
            Self::ARRAY => "array",
            Self::OBJECT => "object",
            _ => return None,
        })
    }
}

const MAX_SCHEMA_REF_DEPTH: usize = 16;
const MAX_SCHEMA_NODES: usize = 1024;

/// The types a declared parameter admits. `None` when the tool or the parameter is not
/// declared, or its schema gives no usable type information.
fn param_types(
    tools: Option<&[ToolDefinition]>,
    function_name: &str,
    param_name: &str,
) -> Option<SchemaTypes> {
    let tool = tools?.iter().find(|t| t.name == function_name)?;
    let root = tool.parameters.as_ref()?;
    let param = root.get("properties")?.get(param_name)?;
    let mut budget = MAX_SCHEMA_NODES;
    schema_types(param, root, 0, &mut budget)
}

fn constrain(types: &mut Option<u8>, by: Option<u8>) {
    if let Some(by) = by {
        *types = Some(types.map_or(by, |types| types & by));
    }
}

/// The types `schema` admits: what its `type`, `enum`, `const`, local `$ref`, `anyOf` and
/// `oneOf` (the union of the branches) and `allOf` (each branch) allow, intersected. A
/// keyword without usable type information (an unknown type name, an untyped branch, a
/// remote or unresolvable reference) does not constrain. `None` when nothing constrains,
/// when the constraints contradict each other, or when the schema is too deep or too
/// large to walk.
fn schema_types(
    schema: &Value,
    root: &Value,
    ref_depth: usize,
    budget: &mut usize,
) -> Option<SchemaTypes> {
    *budget = budget.checked_sub(1)?;
    let schema = schema.as_object()?;
    let mut types = None;

    constrain(
        &mut types,
        match schema.get("type") {
            Some(Value::String(name)) => SchemaTypes::named(name),
            Some(Value::Array(names)) => names.iter().try_fold(0, |acc, name| {
                name.as_str().and_then(SchemaTypes::named).map(|t| acc | t)
            }),
            _ => None,
        },
    );
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        constrain(
            &mut types,
            Some(
                values
                    .iter()
                    .fold(0, |acc, v| acc | SchemaTypes::of_value(v)),
            ),
        );
    }
    if let Some(value) = schema.get("const") {
        constrain(&mut types, Some(SchemaTypes::of_value(value)));
    }
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
        && ref_depth < MAX_SCHEMA_REF_DEPTH
        && let Some(target) = resolve_local_ref(reference, root)
    {
        constrain(
            &mut types,
            schema_types(target, root, ref_depth + 1, budget).map(|t| t.0),
        );
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(branches) = schema.get(keyword).and_then(Value::as_array)
            && !branches.is_empty()
        {
            let union = branches.iter().try_fold(0, |acc, branch| {
                schema_types(branch, root, ref_depth, budget).map(|t| acc | t.0)
            });
            constrain(&mut types, union);
        }
    }
    if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
        for branch in branches {
            constrain(
                &mut types,
                schema_types(branch, root, ref_depth, budget).map(|t| t.0),
            );
        }
    }
    types.filter(|t| *t != 0).map(SchemaTypes)
}

/// A `#` or `#/json/pointer` reference into the tool's parameter schema.
fn resolve_local_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
    match reference.strip_prefix('#')? {
        "" => Some(root),
        pointer => root.pointer(pointer),
    }
}

/// An integer in RFC 8259 grammar: no sign but `-`, no leading zeros.
fn is_json_integer_literal(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    is_integer_literal(text) && (digits == "0" || !digits.starts_with('0'))
}

/// The JSON value `text` spells (RFC 8259, surrounding whitespace allowed), when its type
/// is one of `types` and is not string.
fn admitted_json_value(text: &str, types: SchemaTypes) -> Option<ParsedValue> {
    if is_json_integer_literal(text) {
        // Through the integer coercer, which keeps integers beyond i64 exact.
        return types
            .admits(SchemaTypes::INTEGER)
            .then(|| coerce_integer_literal(text))
            .flatten();
    }
    let value: Value = serde_json::from_str(text).ok()?;
    let value_type = match &value {
        Value::Null => SchemaTypes::NULL,
        Value::Bool(_) => SchemaTypes::BOOLEAN,
        Value::Number(_) => SchemaTypes::NUMBER,
        Value::Array(_) => SchemaTypes::ARRAY,
        Value::Object(_) => SchemaTypes::OBJECT,
        Value::String(_) => return None,
    };
    types.admits(value_type).then(|| value.into())
}

/// Coerce an argument's text to the types its parameter schema admits.
///
/// - One admitted type, however the schema spells it (`"type": "number"`, `["number"]`,
///   an `anyOf` of one type, an `enum` of one type): that type's coercion.
/// - Several admitted types: the JSON value the text spells, when its type is admitted
///   and is not string. Otherwise a parameter that admits string gets the text verbatim,
///   as a `string` parameter does (GLM writes string arguments unquoted and every other
///   value as JSON, so quoted text stays quoted), and one that does not gets the untyped
///   handling. So for `["string", "integer"]`, `42` is 42 and `007`, `1.0` and `+5` stay
///   strings; for `["string", "null"]`, `null` is null. SGLang's GLM-4.7 detector
///   resolves unions the same way; vLLM also prefers the typed value but reads `007`
///   as 7.
/// - No usable type information (an undeclared tool or parameter, `{}`, an unresolvable
///   reference): the untyped handling, which parses only text that is already JSON.
fn coerce_param_value(raw: &str, types: Option<SchemaTypes>) -> ParsedValue {
    let Some(types) = types else {
        return coerce_value(raw, None);
    };
    if let Some(name) = types.single() {
        return coerce_value(raw, Some(name));
    }
    if let Some(value) = admitted_json_value(raw.trim(), types) {
        return value;
    }
    coerce_value(raw, types.admits(SchemaTypes::STRING).then_some("string"))
}

/// Parse a single GLM-4.7 tool call block
/// Format: <tool_call>function_name<arg_key>key1</arg_key><arg_value>value1</arg_value>...</tool_call>
fn parse_tool_call_block(
    block: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<ToolCallResponse> {
    // Remove the outer <tool_call> tags
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    // Strip the outer start token. The end token is optional so we can
    // recover from max_tokens / EOS truncation that drops `</tool_call>`.
    let after_start = block
        .strip_prefix(start_token.as_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid tool call block format"))?;
    let content = after_start
        .strip_suffix(end_token.as_str())
        .unwrap_or(after_start);

    // Extract function name (everything before first <arg_key> or end)
    let arg_key_start = &config.arg_key_start;
    let function_name = if let Some(pos) = content.find(arg_key_start.as_str()) {
        content[..pos].trim().to_string()
    } else {
        // No arguments, just function name
        content.trim().to_string()
    };

    if function_name.is_empty() {
        anyhow::bail!("Empty function name in tool call");
    }

    // Parse key-value pairs, keeping the order the model emitted them in.
    let mut arguments: Vec<(String, ParsedValue)> = Vec::new();
    let mut argument_indices: HashMap<&str, usize> = HashMap::new();
    let args_section = &content[function_name.len()..];

    // Build regex patterns
    let arg_key_start_escaped = regex::escape(&config.arg_key_start);
    let arg_key_end_escaped = regex::escape(&config.arg_key_end);
    let arg_value_start_escaped = regex::escape(&config.arg_value_start);
    let arg_value_end_escaped = regex::escape(&config.arg_value_end);

    // Pattern to match: <arg_key>key</arg_key><arg_value>value</arg_value>
    // (?s) enables dotall mode so (.*?) matches across newlines — required
    // because models often emit multi-line content in arg values.
    let pattern = format!(
        r"(?s){}([^<]+){}{}(.*?){}",
        arg_key_start_escaped, arg_key_end_escaped, arg_value_start_escaped, arg_value_end_escaped
    );

    let regex = Regex::new(&pattern)?;

    for cap in regex.captures_iter(args_section) {
        let key = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let raw_value = cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if !key.is_empty() {
            // Decode XML entities (e.g. &lt; → <, &amp; → &) before parsing
            let decoded = decode_xml_entities(raw_value);

            let types = param_types(tools, &function_name, key);
            let json_value = coerce_param_value(&decoded, types);

            match argument_indices.get(key).copied() {
                Some(index) => arguments[index].1 = json_value,
                None => {
                    argument_indices.insert(key, arguments.len());
                    arguments.push((key.to_string(), json_value));
                }
            }
        }
    }

    // Validate function against tools if provided
    if let Some(tools_list) = tools {
        let tool_exists = tools_list.iter().any(|t| t.name == function_name);
        if !tool_exists {
            anyhow::bail!("Function '{}' not found in available tools", function_name);
        }
    }

    Ok(ToolCallResponse {
        id: Uuid::new_v4().to_string(),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: function_name,
            arguments: serde_json::to_string(&OrderedArguments(&arguments))?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_config() -> Glm47ParserConfig {
        Glm47ParserConfig::default()
    }

    #[test] // helper
    fn test_detect_tool_call_start() {
        let config = get_test_config();

        // Complete start token
        assert!(detect_tool_call_start_glm47(
            "<tool_call>get_weather",
            &config
        ));

        // Partial start token (streaming)
        assert!(detect_tool_call_start_glm47("Some text <tool", &config));
        assert!(detect_tool_call_start_glm47("Some text <tool_c", &config));

        // No tool call
        assert!(!detect_tool_call_start_glm47("Just normal text", &config));
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.1 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.1
    fn test_parse_simple_tool_call() {
        let config = get_test_config();
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args.get("location").unwrap().as_str().unwrap(),
            "San Francisco"
        );
        assert_eq!(normal_text, Some("".to_string()));
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.1, TOOLCALLING.batch.7.d in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.7.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.1, TOOLCALLING.batch.7
    fn test_parse_tool_call_with_multiple_args() {
        let config = get_test_config();
        let message = "<tool_call>book_flight<arg_key>from</arg_key><arg_value>NYC</arg_value><arg_key>to</arg_key><arg_value>LAX</arg_value><arg_key>date</arg_key><arg_value>2026-03-15</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "book_flight");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("from").unwrap().as_str().unwrap(), "NYC");
        assert_eq!(args.get("to").unwrap().as_str().unwrap(), "LAX");
        assert_eq!(args.get("date").unwrap().as_str().unwrap(), "2026-03-15");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.d in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7
    fn test_parse_tool_call_with_json_value() {
        let config = get_test_config();
        let message = r#"<tool_call>search<arg_key>filters</arg_key><arg_value>{"category": "books", "price_max": 50}</arg_value></tool_call>"#;

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let filters = args.get("filters").unwrap();
        assert!(filters.is_object());
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.b in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.2.yaml.
    #[test] // TOOLCALLING.batch.2
    fn test_parse_multiple_tool_calls() {
        let config = get_test_config();
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.8
    fn test_parse_with_normal_text() {
        let config = get_test_config();
        let message = "I'll check the weather for you. <tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            normal_text,
            Some("I'll check the weather for you. ".to_string())
        );
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.6.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.6.yaml.
    #[test] // TOOLCALLING.batch.6
    fn test_parse_tool_call_no_args() {
        let config = get_test_config();
        let message = "<tool_call>get_current_time</tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_current_time");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args.is_empty());
    }

    #[test] // helper
    fn test_find_tool_call_end_position() {
        let config = get_test_config();
        let chunk =
            "<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>more text";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"
        );
    }

    #[test] // helper — parallel calls: end position must advance past ALL blocks
    fn test_find_tool_call_end_position_parallel() {
        let config = get_test_config();
        let chunk = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>SF</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>trailing";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>SF</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
            "Must advance past ALL consecutive tool call blocks"
        );
    }

    #[test] // helper — parallel calls with whitespace between blocks
    fn test_find_tool_call_end_position_parallel_with_whitespace() {
        let config = get_test_config();
        let chunk = "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call>\n<tool_call>b<arg_key>k</arg_key><arg_value>2</arg_value></tool_call>";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            end_pos,
            chunk.len(),
            "Must handle whitespace/newlines between consecutive blocks"
        );
    }

    #[test] // helper — parallel calls: second block incomplete (streaming)
    fn test_find_tool_call_end_position_parallel_second_incomplete() {
        let config = get_test_config();
        let chunk = "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call><tool_call>b<arg_key>k</arg_key><arg_value>2</arg_value>";

        let end_pos = find_tool_call_end_position_glm47(chunk, &config);
        assert_eq!(
            &chunk[..end_pos],
            "<tool_call>a<arg_key>k</arg_key><arg_value>1</arg_value></tool_call>",
            "Must stop at first complete block when second is incomplete"
        );
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.c, TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.2.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.2 + TOOLCALLING.batch.8 — bug report repro: text + parallel calls
    fn test_parse_text_then_parallel_calls() {
        let config = get_test_config();
        let message = "I'll check the weather for both cities at the same time!<tool_call>get_weather<arg_key>location</arg_key><arg_value>San Francisco</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>New York</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2, "Both parallel calls must be extracted");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");

        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(
            args0.get("location").unwrap().as_str().unwrap(),
            "San Francisco"
        );
        assert_eq!(args1.get("location").unwrap().as_str().unwrap(), "New York");

        let text = normal_text.unwrap();
        assert_eq!(
            text,
            "I'll check the weather for both cities at the same time!"
        );
    }

    #[test]
    fn test_parse_two_bare_calls_without_start_markers_recovers_independently() {
        let config = get_test_config();
        let message = "I will check both. get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>";

        let (calls, normal_text) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0["location"], "NYC");
        assert_eq!(args1["timezone"], "EST");
        assert_eq!(normal_text.unwrap(), "I will check both. ");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.b in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.7.yaml.
    #[test] // TOOLCALLING.batch.7, TOOLCALLING.fmt.2
    fn test_parse_multiline_arg_value() {
        let config = get_test_config();
        let message = "<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/hello.py</arg_value><arg_key>content</arg_key><arg_value>#!/usr/bin/env python3\nprint(\"Hello, World!\")\n</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "write_file");

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args.get("path").unwrap().as_str().unwrap(), "/tmp/hello.py");
        assert!(
            args.contains_key("content"),
            "content argument must be parsed even when it contains newlines"
        );
        let content = args.get("content").unwrap().as_str().unwrap();
        assert!(content.contains("print(\"Hello, World!\")"));
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.4.d in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.4.yaml.
    #[test] // TOOLCALLING.batch.4
    fn test_malformed_tool_call() {
        let config = get_test_config();

        // Missing end tag
        let message = "<tool_call>get_weather";
        let result = try_tool_call_parse_glm47(message, &config, None);
        assert!(result.is_ok()); // Should handle gracefully, no calls extracted

        let (calls, _) = result.unwrap();
        assert_eq!(calls.len(), 0);
    }

    // Recovery for missing outer </tool_call> (max_tokens / EOS truncation):
    // when the inner arg pairs are well-formed, treat EOF as the end token
    // and extract the call. The arg_key opener gates recovery so plain text
    // that happens to start with `<tool_call>` is still preserved verbatim.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.5.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.5.yaml.
    #[test] // TOOLCALLING.batch.5
    fn test_parse_no_end_tag_complete_args_recovers() {
        let config = Glm47ParserConfig {
            allow_eof_recovery: true,
            ..get_test_config()
        };
        // Args complete, only outer </tool_call> missing.
        let message = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "NYC");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.b, TOOLCALLING.batch.5.a in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.2.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.5.yaml.
    #[test] // TOOLCALLING.batch.5
    fn test_parse_no_end_tag_multiple_calls_recovers() {
        let config = Glm47ParserConfig {
            allow_eof_recovery: true,
            ..get_test_config()
        };
        // Two complete inner calls, missing only the trailing </tool_call> on the second.
        let message = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_time<arg_key>tz</arg_key><arg_value>EST</arg_value>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.8.c, TOOLCALLING.batch.13 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.13.yaml, tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.8.yaml.
    #[test] // TOOLCALLING.batch.4, TOOLCALLING.batch.8
    fn test_unparseable_block_dropped_no_tag_leak() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];

        // Tool call block references a function not in the tools list — the
        // whole block (including <tool_call>...<arg_key>...<arg_value>... wire
        // markup) must be dropped, not leaked through normal_text.
        let message = "Here is the result: <tool_call>unknown_func<arg_key>x</arg_key><arg_value>1</arg_value></tool_call> done";
        let (calls, normal_text) =
            try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        assert_eq!(calls.len(), 0);
        let text = normal_text.unwrap();
        assert!(
            !text.contains("unknown_func"),
            "Unparseable block must be dropped to avoid tag leakage, got: {text}"
        );
        assert!(
            !text.contains("<tool_call>") && !text.contains("<arg_key>"),
            "Wire-format tags must not leak into normal_text, got: {text}"
        );
        assert!(
            text.contains("Here is the result:") && text.contains("done"),
            "Surrounding prose must be preserved, got: {text}"
        );
    }

    #[test] // helper
    fn test_xml_entity_decoding() {
        let config = get_test_config();
        let message = r#"<tool_call>write_file<arg_key>content</arg_key><arg_value>x &lt; y &amp;&amp; y &gt; z</arg_value></tool_call>"#;

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args.get("content").unwrap().as_str().unwrap(),
            "x < y && y > z"
        );
    }

    #[test] // helper
    fn test_type_coercion_with_schema() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "set_temperature".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "degrees": {"type": "number"},
                    "enabled": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "huge_integer": {"type": "integer"},
                    "large_count": {"type": "number"},
                    "huge_count": {"type": "number"},
                    "label": {"type": "string"}
                }
            })),
            strict: None,
        }];

        let message = "<tool_call>set_temperature<arg_key>degrees</arg_key><arg_value>72.5</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>count</arg_key><arg_value>3</arg_value><arg_key>huge_integer</arg_key><arg_value>9223372036854775808</arg_value><arg_key>large_count</arg_key><arg_value>9007199254740993</arg_value><arg_key>huge_count</arg_key><arg_value>100000000000000000000</arg_value><arg_key>label</arg_key><arg_value>warm</arg_value></tool_call>";

        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();

        // number coercion
        assert_eq!(args.get("degrees").unwrap().as_f64().unwrap(), 72.5);
        // boolean coercion
        assert!(args.get("enabled").unwrap().as_bool().unwrap());
        // integer coercion
        assert_eq!(args.get("count").unwrap().as_i64().unwrap(), 3);
        // integer-like numbers should not be rounded through f64
        assert_eq!(
            args.get("large_count").unwrap().as_i64().unwrap(),
            9007199254740993
        );
        let raw_args: HashMap<String, Box<serde_json::value::RawValue>> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(raw_args["huge_integer"].get(), "9223372036854775808");
        assert_eq!(raw_args["huge_count"].get(), "100000000000000000000");
        // string stays string
        assert_eq!(args.get("label").unwrap().as_str().unwrap(), "warm");
    }

    #[test]
    fn test_arguments_serialized_in_source_order() {
        let config = get_test_config();
        let message = concat!(
            "<tool_call>create_ticket",
            "<arg_key>title</arg_key><arg_value>Rotate keys</arg_value>",
            "<arg_key>description</arg_key><arg_value>Keys are 90 days old</arg_value>",
            "<arg_key>priority</arg_key><arg_value>medium</arg_value>",
            "<arg_key>labels</arg_key><arg_value>[\"security\", \"ops\"]</arg_value>",
            "<arg_key>title</arg_key><arg_value>Rotate S3 keys</arg_value>",
            "</tool_call>"
        );

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        let keys: Vec<&str> = regex::Regex::new(r#""([a-z_]+)":"#)
            .unwrap()
            .captures_iter(&calls[0].function.arguments)
            .map(|c| c.get(1).unwrap().as_str())
            .collect();
        assert_eq!(keys, ["title", "description", "priority", "labels"]);
        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["title"], Value::String("Rotate S3 keys".to_string()));
        assert_eq!(args["labels"], serde_json::json!(["security", "ops"]));
    }

    #[test]
    fn test_many_arguments_preserve_order_and_replace_duplicates() {
        const COUNT: usize = 4096;
        let replacements = [COUNT - 1, COUNT / 2, 0];
        let mut message = String::from("<tool_call>bulk_update");
        for i in (0..COUNT).rev() {
            message.push_str(&format!(
                "<arg_key>arg_{i}</arg_key><arg_value>initial</arg_value>"
            ));
        }
        for i in replacements {
            message.push_str(&format!(
                "<arg_key>arg_{i}</arg_key><arg_value>updated</arg_value>"
            ));
        }
        message.push_str("</tool_call>");

        let (calls, _) = try_tool_call_parse_glm47(&message, &get_test_config(), None).unwrap();
        assert_eq!(calls.len(), 1);
        let expected = (0..COUNT)
            .rev()
            .map(|i| {
                let value = if replacements.contains(&i) {
                    "updated"
                } else {
                    "initial"
                };
                format!(r#""arg_{i}":"{value}""#)
            })
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(calls[0].function.arguments, format!("{{{expected}}}"));
    }

    #[test]
    fn test_string_schema_keeps_json_looking_values_verbatim() {
        for param_schema in [
            serde_json::json!({"type": "string"}),
            serde_json::json!({"type": ["string"]}),
            serde_json::json!({"type": ["string", "null"]}),
            serde_json::json!({"type": ["null", "string"]}),
            serde_json::json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
            serde_json::json!({"oneOf": [{"type": "null"}, {"type": "string"}]}),
            serde_json::json!({"allOf": [{"type": "string"}]}),
            serde_json::json!({"allOf": [{"minLength": 1}, {"type": "string"}]}),
            serde_json::json!({"oneOf": [
                {"type": "null"}, {"allOf": [{"type": "string"}, {"minLength": 1}]}
            ]}),
        ] {
            let config = get_test_config();
            let tools = vec![ToolDefinition {
                name: "save_note".to_string(),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "object_text": param_schema,
                        "array_text": param_schema,
                        "quoted_text": param_schema,
                        "payload": {"type": "object"},
                        "composed_payload": {"allOf": [
                            {"type": "object"},
                            {"properties": {"key": {"type": "string"}}}
                        ]},
                        "untyped": {}
                    }
                })),
                strict: None,
            }];

            let message = concat!(
                "<tool_call>save_note",
                "<arg_key>object_text</arg_key><arg_value>{\"key\": \"value\"}</arg_value>",
                "<arg_key>array_text</arg_key><arg_value>[1, 2, 3]</arg_value>",
                "<arg_key>quoted_text</arg_key><arg_value>\"quoted\"</arg_value>",
                "<arg_key>payload</arg_key><arg_value>{\"key\": \"value\"}</arg_value>",
                "<arg_key>composed_payload</arg_key><arg_value>{\"key\": \"value\"}</arg_value>",
                "<arg_key>untyped</arg_key><arg_value>[1, 2, 3]</arg_value>",
                "</tool_call>"
            );

            let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();
            assert_eq!(calls.len(), 1);
            let args: HashMap<String, Value> =
                serde_json::from_str(&calls[0].function.arguments).unwrap();

            assert_eq!(
                args["object_text"],
                Value::String("{\"key\": \"value\"}".to_string())
            );
            assert_eq!(args["array_text"], Value::String("[1, 2, 3]".to_string()));
            assert_eq!(args["quoted_text"], Value::String("\"quoted\"".to_string()));
            assert_eq!(args["payload"], serde_json::json!({"key": "value"}));
            assert_eq!(
                args["composed_payload"],
                serde_json::json!({"key": "value"})
            );
            assert_eq!(args["untyped"], serde_json::json!([1, 2, 3]));
        }
    }

    fn single_param(schema: Value) -> Value {
        serde_json::json!({"type": "object", "properties": {"v": schema}})
    }

    /// The `v` argument parsed from `text` for a tool `f` with these parameters.
    fn parse_arg(parameters: &Value, text: &str) -> Value {
        let tools = vec![ToolDefinition {
            name: "f".to_string(),
            parameters: Some(parameters.clone()),
            strict: None,
        }];
        let message =
            format!("<tool_call>f<arg_key>v</arg_key><arg_value>{text}</arg_value></tool_call>");
        let (calls, _) =
            try_tool_call_parse_glm47(&message, &get_test_config(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1, "{message}");
        let mut args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        args.remove("v").unwrap()
    }

    #[test]
    fn test_single_type_is_coerced_however_the_schema_spells_it() {
        use serde_json::json;
        for (schema, text, expected) in [
            (
                json!({"type": ["number"], "minimum": -10}),
                "-10",
                json!(-10),
            ),
            (json!({"type": ["integer"]}), "0", json!(0)),
            (json!({"type": ["boolean"]}), "false", json!(false)),
            (json!({"type": ["null"]}), "null", Value::Null),
            (
                json!({"anyOf": [{"type": "boolean"}, {"type": "boolean"}]}),
                "true",
                json!(true),
            ),
            (
                json!({"anyOf": [{"type": "null"}, {"type": "null"}]}),
                "null",
                Value::Null,
            ),
            (
                json!({"oneOf": [{"type": "integer", "minimum": 0}, {"type": "integer", "maximum": -5}]}),
                "7",
                json!(7),
            ),
            (json!({"enum": [1, 2, 3]}), "2", json!(2)),
            (json!({"enum": [1.5, 2.5]}), "2.5", json!(2.5)),
            (json!({"const": true}), "true", json!(true)),
            (
                json!({"type": "number", "enum": [1.23, 4.56]}),
                "4.56",
                json!(4.56),
            ),
            (
                json!({"allOf": [{"type": ["integer", "string"]}, {"type": "integer"}]}),
                "12",
                json!(12),
            ),
            // One type keeps that type's lenient coercion.
            (json!({"anyOf": [{"type": "boolean"}]}), "yes", json!(true)),
            (json!({"type": ["integer"]}), "007", json!(7)),
        ] {
            assert_eq!(
                parse_arg(&single_param(schema.clone()), text),
                expected,
                "{schema} {text}"
            );
        }
    }

    #[test]
    fn test_local_refs_give_the_parameter_type() {
        use serde_json::json;
        let parameters = json!({
            "type": "object",
            "properties": {"v": {"$ref": "#/$defs/Count"}},
            "$defs": {"Count": {"type": "integer"}}
        });
        assert_eq!(parse_arg(&parameters, "5"), json!(5));

        let parameters = json!({
            "type": "object",
            "properties": {"v": {"$ref": "#/$defs/Node"}},
            "$defs": {"Node": {
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/Node"}}
            }}
        });
        assert_eq!(
            parse_arg(&parameters, "{\"next\": {}}"),
            json!({"next": {}})
        );

        // A string definition keeps JSON-looking text as written.
        let parameters = json!({
            "type": "object",
            "properties": {"v": {"$ref": "#/$defs/Name"}},
            "$defs": {"Name": {"type": "string"}}
        });
        assert_eq!(parse_arg(&parameters, "{\"a\": 1}"), json!("{\"a\": 1}"));

        // `#` is the parameters schema itself.
        let parameters = json!({"type": "object", "properties": {"v": {"$ref": "#"}}});
        assert_eq!(parse_arg(&parameters, "{\"v\": 1}"), json!({"v": 1}));
    }

    #[test]
    fn test_union_takes_the_json_value_of_an_admitted_type() {
        use serde_json::json;
        for (schema, text, expected) in [
            (json!({"type": ["string", "null"]}), "null", Value::Null),
            (json!({"type": ["null", "string"]}), " null ", Value::Null),
            (
                json!({"type": ["string", "null"], "enum": ["value", null]}),
                "null",
                Value::Null,
            ),
            (
                json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
                "null",
                Value::Null,
            ),
            (json!({"type": ["string", "integer"]}), "42", json!(42)),
            (json!({"type": ["string", "integer"]}), "-7", json!(-7)),
            (json!({"type": ["string", "number"]}), "1.5", json!(1.5)),
            (json!({"type": ["string", "number"]}), "1e3", json!(1000.0)),
            (json!({"type": ["string", "boolean"]}), "true", json!(true)),
            (
                json!({"type": ["string", "array"]}),
                "[\"a\", \"b\"]",
                json!(["a", "b"]),
            ),
            (
                json!({"anyOf": [{"type": "string"}, {"type": "object"}]}),
                "{\"k\": 1}",
                json!({"k": 1}),
            ),
            (json!({"type": ["integer", "null"]}), "null", Value::Null),
            (json!({"type": ["integer", "null"]}), "-3", json!(-3)),
            (json!({"type": ["boolean", "null"]}), "false", json!(false)),
            (
                json!({"anyOf": [{"type": "array"}, {"type": "null"}]}),
                "[1, 2]",
                json!([1, 2]),
            ),
            // An integer in JSON is an integer, not the boolean `1`.
            (json!({"type": ["integer", "boolean"]}), "1", json!(1)),
            (json!({"enum": ["a", 1, null]}), "1", json!(1)),
        ] {
            assert_eq!(
                parse_arg(&single_param(schema.clone()), text),
                expected,
                "{schema} {text}"
            );
        }
    }

    #[test]
    fn test_union_keeps_integers_beyond_i64_exact() {
        let tools = vec![ToolDefinition {
            name: "f".to_string(),
            parameters: Some(single_param(
                serde_json::json!({"type": ["string", "integer"]}),
            )),
            strict: None,
        }];
        let message = "<tool_call>f<arg_key>v</arg_key><arg_value>123456789012345678901234567890</arg_value></tool_call>";
        let (calls, _) =
            try_tool_call_parse_glm47(message, &get_test_config(), Some(&tools)).unwrap();
        assert_eq!(
            calls[0].function.arguments,
            r#"{"v":123456789012345678901234567890}"#
        );
    }

    #[test]
    fn test_union_with_string_keeps_other_text_verbatim() {
        use serde_json::json;
        for (schema, text) in [
            // Not JSON: leading zeros, a plus sign, Python spellings, bare fractions.
            (json!({"type": ["string", "integer"]}), "007"),
            (json!({"type": ["string", "integer"]}), "+5"),
            (json!({"type": ["string", "boolean"]}), "True"),
            (json!({"type": ["string", "boolean"]}), "yes"),
            (json!({"type": ["string", "null"]}), "None"),
            (json!({"type": ["string", "null"]}), ""),
            (json!({"type": ["string", "number"]}), ".5"),
            (json!({"type": ["string", "number"]}), "NaN"),
            (json!({"type": ["string", "integer"]}), "1 2"),
            // JSON, but of a type the schema does not admit.
            (json!({"type": ["string", "integer"]}), "1.0"),
            (json!({"type": ["string", "integer"]}), "[1, 2]"),
            (json!({"type": ["string", "null"]}), "{\"k\": 1}"),
            // Quoted text is a string the model wrote with its quotes.
            (json!({"type": ["string", "integer"]}), "\"42\""),
            (json!({"type": ["string", "integer"]}), "  two words  "),
        ] {
            assert_eq!(
                parse_arg(&single_param(schema.clone()), text),
                json!(text),
                "{schema} {text}"
            );
        }
    }

    #[test]
    fn test_union_without_string_falls_back_to_untyped_handling() {
        use serde_json::json;
        let schema = single_param(json!({"type": ["integer", "null"]}));
        assert_eq!(parse_arg(&schema, "abc"), json!("abc"));
        assert_eq!(parse_arg(&schema, "1.5"), json!("1.5"));
        assert_eq!(parse_arg(&schema, "[1]"), json!([1]));
    }

    #[test]
    fn test_untyped_or_unresolvable_schema_keeps_untyped_handling() {
        use serde_json::json;
        for schema in [
            json!({}),
            json!({"description": "anything"}),
            json!({"type": "any"}),
            json!({"$ref": "#/$defs/missing"}),
            json!({"$ref": "https://example.com/schema.json"}),
            json!({"anyOf": [{"type": "integer"}, {}]}),
            json!({"allOf": [{"type": "string"}, {"type": "integer"}]}),
        ] {
            let parameters = single_param(schema.clone());
            assert_eq!(parse_arg(&parameters, "42"), json!("42"), "{schema}");
            assert_eq!(parse_arg(&parameters, "[1, 2]"), json!([1, 2]), "{schema}");
        }
    }

    #[test]
    fn test_reference_cycles_are_bounded() {
        use serde_json::json;
        for defs in [
            json!({"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}),
            json!({"A": {"anyOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/A"}]}}),
            json!({"A": {"allOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/A"}]}}),
        ] {
            let parameters = json!({
                "type": "object",
                "properties": {"v": {"$ref": "#/$defs/A"}},
                "$defs": defs
            });
            assert_eq!(parse_arg(&parameters, "42"), json!("42"), "{parameters}");
        }
    }

    #[test]
    fn test_string_union_takes_json_of_an_admitted_container_type() {
        use serde_json::json;
        for (schema, object_admitted, array_admitted) in [
            (json!({"type": ["object", "array", "string"]}), true, true),
            (
                json!({"allOf": [
                    {"anyOf": [
                        {"type": "null"},
                        {"oneOf": [{"type": "array"}, {"type": "string"}]}
                    ]},
                    {"minLength": 1}
                ]}),
                false,
                true,
            ),
            (
                json!({"anyOf": [
                    {"type": "object"},
                    {"oneOf": [{"type": "array"}, {"type": ["null", "string"]}]}
                ]}),
                true,
                true,
            ),
        ] {
            let parameters = single_param(schema.clone());
            let object_text = "{\"key\": \"value\"}";
            let array_text = "[1, 2, 3]";
            assert_eq!(
                parse_arg(&parameters, object_text),
                if object_admitted {
                    json!({"key": "value"})
                } else {
                    json!(object_text)
                },
                "{schema}"
            );
            assert_eq!(
                parse_arg(&parameters, array_text),
                if array_admitted {
                    json!([1, 2, 3])
                } else {
                    json!(array_text)
                },
                "{schema}"
            );
            assert_eq!(
                parse_arg(&parameters, "\"quoted\""),
                json!("\"quoted\""),
                "{schema}"
            );
        }
    }

    #[test] // helper
    fn test_type_coercion_array_comma_separated() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "tag_item".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "tags": {"type": "array"}
                }
            })),
            strict: None,
        }];

        // Model emits comma-separated values without JSON brackets
        let message = "<tool_call>tag_item<arg_key>tags</arg_key><arg_value>rust, python, go</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let tags = args.get("tags").unwrap().as_array().unwrap();
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0].as_str().unwrap(), "rust");
        assert_eq!(tags[1].as_str().unwrap(), "python");
        assert_eq!(tags[2].as_str().unwrap(), "go");
    }

    #[test] // helper
    fn test_type_coercion_array_json() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "tag_item".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "ids": {"type": "array"}
                }
            })),
            strict: None,
        }];

        // Model emits proper JSON array
        let message = r#"<tool_call>tag_item<arg_key>ids</arg_key><arg_value>[1, 2, 3]</arg_value></tool_call>"#;
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let ids = args.get("ids").unwrap().as_array().unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0].as_i64().unwrap(), 1);
    }

    #[test] // helper
    fn test_type_coercion_falls_back_to_string() {
        let config = get_test_config();
        let tools = vec![ToolDefinition {
            name: "test_func".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"}
                }
            })),
            strict: None,
        }];

        // "not_a_number" can't be parsed as integer — should fall back to string
        let message = "<tool_call>test_func<arg_key>count</arg_key><arg_value>not_a_number</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(
            args.get("count").unwrap().is_string(),
            "Should fall back to string when coercion fails"
        );
    }

    /// Parser-level invariant: the glm47 parser is byte-stable — it doesn't
    /// see `finish_reason` and produces the same output regardless of the
    /// upstream stream-end reason. Real PIPELINE.finish_reason coverage (stop / tool_calls
    /// / length mapping) lives in `lib/llm/tests/test_streaming_tool_parsers.rs`
    /// and belongs in the cross-parser finish_reason mapping work-item
    /// (tracked separately).
    #[test]
    fn test_glm47_parser_output_independent_of_upstream_finish() {
        let config = get_test_config();
        let input = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
    }

    /// TOOLCALLING.batch.9 — empty / null content variants. Truly-empty (zero bytes)
    /// and whitespace-only inputs must yield no tool calls; normal_text
    /// collapses to the empty string.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.9 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.9
    fn test_parse_glm47_empty_and_whitespace_inputs() {
        let config = get_test_config();
        for input in &["", " ", "\n", "\t\n  \t"] {
            let (calls, normal) = try_tool_call_parse_glm47(input, &config, None).unwrap();
            assert!(
                calls.is_empty(),
                "Empty/whitespace input must yield no calls (input={:?})",
                input
            );
            assert_eq!(
                normal.as_deref(),
                Some(""),
                "Empty/whitespace input collapses to empty normal_text (input={:?})",
                input
            );
        }
    }

    /// TOOLCALLING.batch.10 — duplicate calls (same function name twice in one section).
    /// Universal gap noted in the test taxonomy; pin parser-level behavior —
    /// both calls returned with distinct ids.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.10 in tests/parity/toolcalling/fixtures/glm47/TOOLCALLING.batch.yaml.
    #[test] // TOOLCALLING.batch.10
    fn test_parse_glm47_duplicate_calls_same_name() {
        let config = get_test_config();
        let input = "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call><tool_call>get_weather<arg_key>location</arg_key><arg_value>LA</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(input, &config, None).unwrap();
        assert_eq!(calls.len(), 2, "Both duplicate-name calls must be returned");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");
        assert_ne!(
            calls[0].id, calls[1].id,
            "Duplicate calls must have distinct ids"
        );
        let args0: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: HashMap<String, Value> =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0.get("location").unwrap().as_str().unwrap(), "NYC");
        assert_eq!(args1.get("location").unwrap().as_str().unwrap(), "LA");
    }
}

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

/// Why a GLM-4.7 tool_call block failed to parse. No message quotes the block:
/// [`BlockError::kind`] is the label used in logs, and callers may log the
/// message too.
#[derive(Debug)]
pub(crate) enum BlockError {
    InvalidFormat,
    EmptyFunctionName,
    UndeclaredNonIdentifier,
    Pattern(regex::Error),
    Arguments(serde_json::Error),
}

impl BlockError {
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            BlockError::InvalidFormat => "invalid_block_format",
            BlockError::EmptyFunctionName => "empty_function_name",
            BlockError::UndeclaredNonIdentifier => "undeclared_non_identifier",
            BlockError::Pattern(_) => "argument_pattern",
            BlockError::Arguments(_) => "argument_serialization",
        }
    }
}

impl std::fmt::Display for BlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockError::InvalidFormat => f.write_str("Invalid tool call block format"),
            BlockError::EmptyFunctionName => f.write_str("Empty function name in tool call"),
            BlockError::UndeclaredNonIdentifier => {
                f.write_str("Function name is not declared and is not an identifier")
            }
            BlockError::Pattern(e) => e.fmt(f),
            BlockError::Arguments(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for BlockError {}

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

/// Whether text that starts at a bare call's first marker can be a call body: the name
/// before it is followed by `<arg_key>` or `</tool_call>`.
fn starts_bare_call_body(from_marker: &str, config: &Glm47ParserConfig) -> bool {
    from_marker.starts_with(config.arg_key_start.as_str())
        || from_marker.starts_with(config.tool_call_end.as_str())
}

fn find_bare_glm47_tool_call_end_position(text: &str, config: &Glm47ParserConfig) -> Option<usize> {
    let marker_idx = first_orphan_glm47_marker_index(text, config)?;
    if !starts_bare_call_body(&text[marker_idx..], config) {
        return None;
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
            dropped_block_len = text.len() - marker_idx,
            marker_offset = marker_idx,
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

            // Read the call with the declared tools, so markup inside an argument
            // value stays in the value. A call this reading cannot place falls
            // through to the first-end-marker handling below.
            let scanner = Glm47Scanner::new(text, config, tools, true);
            if let Scan::Call(call) = scanner.scan_call(abs_start) {
                if scanner.stands_as_call(&call) == Some(false) {
                    if calls.is_empty() {
                        normal_parts.push(text[abs_start..call.end].to_string());
                    }
                    cursor = call.end;
                    continue;
                }
                match build_scanned_call(text, &call, tools) {
                    Ok(parsed_call) => calls.push(parsed_call),
                    Err(e) => {
                        warn!(
                            reason = e.kind(),
                            why = "block read by the schema-aware scan failed to build \
                                   as a GLM-4.7 tool call; dropping to avoid leaking wire \
                                   tags through normal_text",
                            dropped_block_len = call.end - abs_start,
                            block_offset = abs_start,
                            "GLM-4.7 parser dropping unparseable tool_call block"
                        );
                    }
                }
                cursor = call.end;
                continue;
            }

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
                            reason = e.kind(),
                            why = "block has open + close fence but content failed to parse \
                                   as a GLM-4.7 tool call (e.g. empty function name, \
                                   missing <arg_key>, malformed args); dropping to avoid \
                                   leaking wire tags through normal_text",
                            dropped_block_len = block.len(),
                            block_offset = abs_start,
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
                                reason = e.kind(),
                                why = "EOF recovery enabled and <arg_key> opener present, \
                                       but parse_tool_call_block failed on the truncated \
                                       tail; dropping to avoid leaking wire tags through \
                                       normal_text",
                                dropped_block_len = block.len(),
                                block_offset = abs_start,
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
                        dropped_block_len = block.len(),
                        block_offset = abs_start,
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
    // A call name is followed by `<arg_key>` or `</tool_call>`. Text that runs into
    // `</arg_value>` or `</arg_key>` first is the tail of a value or key whose call
    // opener was lost, not a call body.
    if !starts_bare_call_body(&text[marker_idx..], config) {
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
                dropped_block_len = text.len() - cursor,
                recovered_calls = calls.len(),
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

/// Escape raw control characters that appear inside JSON string literals.
fn escape_control_chars_in_strings(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let (mut in_string, mut escaped) = (false, false);
    for c in s.chars() {
        if in_string {
            if escaped {
                escaped = false;
                out.push(c);
            } else if c == '\\' {
                escaped = true;
                out.push(c);
            } else if c == '"' {
                in_string = false;
                out.push(c);
            } else if (c as u32) < 0x20 {
                match c {
                    '\n' => out.push_str("\\n"),
                    '\t' => out.push_str("\\t"),
                    '\r' => out.push_str("\\r"),
                    _ => out.push_str(&format!("\\u{:04x}", c as u32)),
                }
            } else {
                out.push(c);
            }
        } else {
            if c == '"' {
                in_string = true;
            }
            out.push(c);
        }
    }
    out
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
            // Models emit raw tabs/newlines inside JSON strings; strict JSON rejects them.
            if let Ok(v) = serde_json::from_str::<Value>(&escape_control_chars_in_strings(trimmed))
                && v.is_array()
            {
                return v.into();
            }
            // Bracketed text that still fails to parse is broken JSON, not a comma list:
            // splitting it would hand the caller shards of the model's text.
            if trimmed.starts_with('[') {
                return Value::String(raw.to_string()).into();
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

/// Whether an undeclared function name is a plausible identifier: an ASCII letter or `_`,
/// then up to 127 ASCII letters, digits or `_ . : -`. Markup and prose are not names.
fn is_plausible_tool_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && name.len() <= 128
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-'))
}

/// The six GLM-4.7 tool-call tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tag {
    CallStart,
    CallEnd,
    KeyStart,
    KeyEnd,
    ValueStart,
    ValueEnd,
}

impl Tag {
    const ALL: [Tag; 6] = [
        Tag::CallStart,
        Tag::CallEnd,
        Tag::KeyStart,
        Tag::KeyEnd,
        Tag::ValueStart,
        Tag::ValueEnd,
    ];

    fn text(self, config: &Glm47ParserConfig) -> &str {
        match self {
            Tag::CallStart => &config.tool_call_start,
            Tag::CallEnd => &config.tool_call_end,
            Tag::KeyStart => &config.arg_key_start,
            Tag::KeyEnd => &config.arg_key_end,
            Tag::ValueStart => &config.arg_value_start,
            Tag::ValueEnd => &config.arg_value_end,
        }
    }
}

/// How strongly a tag sequence after an `</arg_value>` reads as structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Strength {
    /// A declared, not yet given parameter, or the end of the call followed by
    /// the end of the output or by a call to a declared tool.
    Strong,
    /// A parameter name the schema does not declare or that the call already
    /// gave, or a following call to an undeclared tool.
    Weak,
}

/// What follows a candidate `</arg_value>`.
enum Continuation {
    Structural(Strength),
    /// Not a structural continuation: the tag is text inside the value.
    Literal,
    /// The text so far cannot tell (only while streaming).
    Undecided,
}

/// The outcome of reading one call that starts with `<tool_call>`.
enum Scan {
    Call(ScannedCall),
    /// More output is needed to decide (only while streaming).
    Undecided,
    /// Not a call this grammar reads; the caller falls back to the plain reading.
    Unparsed,
}

struct ScannedCall {
    name: String,
    /// Whether the request declares the tool.
    declared: bool,
    args: Vec<(String, std::ops::Range<usize>)>,
    /// Byte offset just past the call's `</tool_call>`.
    end: usize,
}

/// Literal tag pairs inside an argument value. Markup a value holds is usually written
/// whole (a heredoc that writes a GLM tool call, a test fixture), so an `</arg_value>`
/// reached with every literal pair closed is preferred as the end of the value.
#[derive(Default)]
struct LiteralBalance {
    calls: usize,
    values: usize,
    unmatched_close: bool,
}

impl LiteralBalance {
    fn add(&mut self, tag: Tag) {
        let (open, count) = match tag {
            Tag::CallStart => (true, &mut self.calls),
            Tag::CallEnd => (false, &mut self.calls),
            Tag::ValueStart => (true, &mut self.values),
            Tag::ValueEnd => (false, &mut self.values),
            Tag::KeyStart | Tag::KeyEnd => return,
        };
        if open {
            *count += 1;
        } else if *count == 0 {
            self.unmatched_close = true;
        } else {
            *count -= 1;
        }
    }

    fn closed(&self) -> bool {
        self.calls == 0 && self.values == 0 && !self.unmatched_close
    }
}

/// Reads GLM-4.7 calls with the request's tool schemas.
///
/// GLM writes a string argument as raw text, and its tokenizer turns the text of each
/// tag into the tag's own token, so a value can hold the model's own markup: a heredoc
/// that writes a tool-call fixture, a parser test, a grep for `</tool_call>`. Matching the
/// first `</arg_value>` or `</tool_call>` cuts such a value short and leaks the rest as
/// content. Here a tag inside a value is structure only where the grammar and the
/// declared tools allow it:
///
/// - `</arg_value>` ends a value only if, after optional whitespace, it is followed by
///   `<arg_key>NAME</arg_key><arg_value>` or by `</tool_call>`, and `</tool_call>` in turn
///   by optional text with no tag in it and then the end of the output or another
///   `<tool_call>NAME`;
/// - a following `<arg_key>` is strong when it names a declared parameter the call has not
///   given yet (any identifier when the tool has no declared parameters), and weak when the
///   name is undeclared or repeated; a following call is strong when its tool is declared,
///   or when the request declares no tools;
/// - among the candidates, the first strong one reached with every literal tag pair in the
///   value closed wins; then the first weak one with the pairs closed; then the first strong
///   one; then the first weak one. Every other tag is text inside the value.
///
/// So `<arg_value>cat <<'EOF'\n<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>\nEOF</arg_value></tool_call>`
/// is one call whose value runs to the last `</arg_value>`, while two calls written back to
/// back stay two calls.
///
/// While streaming (`at_eof == false`), a value end is decided only when no later text
/// could change the reading: it is the first candidate of the strongest class, and the text
/// after it already shows the next declared parameter or the next call to a declared tool.
/// Otherwise the scan reports [`Scan::Undecided`] and the caller holds the text until more
/// arrives or the stream ends.
struct Glm47Scanner<'a> {
    text: &'a str,
    config: &'a Glm47ParserConfig,
    tools: Option<&'a [ToolDefinition]>,
    at_eof: bool,
}

impl<'a> Glm47Scanner<'a> {
    fn new(
        text: &'a str,
        config: &'a Glm47ParserConfig,
        tools: Option<&'a [ToolDefinition]>,
        at_eof: bool,
    ) -> Self {
        Self {
            text,
            config,
            tools,
            at_eof,
        }
    }

    fn skip_ws(&self, mut pos: usize) -> usize {
        let rest = &self.text[pos..];
        pos += rest.len() - rest.trim_start().len();
        pos
    }

    /// The first tag at or after `from`, and where it starts.
    fn next_tag(&self, from: usize) -> Option<(usize, Tag)> {
        let mut pos = from;
        while let Some(off) = self.text[pos..].find('<') {
            let at = pos + off;
            let rest = &self.text[at..];
            if let Some(tag) = Tag::ALL
                .into_iter()
                .find(|tag| rest.starts_with(tag.text(self.config)))
            {
                return Some((at, tag));
            }
            pos = at + 1;
        }
        None
    }

    /// Whether `tag` starts at `pos`: `Some(true)`, `Some(false)`, or `None` when the text
    /// ends inside a prefix of it while more may arrive.
    fn tag_at(&self, pos: usize, tag: Tag) -> Option<bool> {
        let rest = &self.text[pos..];
        let tag = tag.text(self.config);
        if rest.starts_with(tag) {
            Some(true)
        } else if !self.at_eof && tag.starts_with(rest) {
            None
        } else {
            Some(false)
        }
    }

    fn tool(&self, name: &str) -> Option<&'a ToolDefinition> {
        self.tools?.iter().find(|t| t.name == name)
    }

    /// Whether the request declares any tool. Without declarations every plausible name
    /// reads as a tool name: there is nothing to tell a fixture from a call by.
    fn declares_tools(&self) -> bool {
        self.tools.is_some_and(|tools| !tools.is_empty())
    }

    fn name_strength(&self, name: &str) -> Option<Strength> {
        if self.tool(name).is_some() {
            Some(Strength::Strong)
        } else if is_plausible_tool_name(name) {
            Some(if self.declares_tools() {
                Strength::Weak
            } else {
                Strength::Strong
            })
        } else {
            None
        }
    }

    fn key_strength(&self, tool: &str, key: &str, given: &[&str]) -> Option<Strength> {
        let declared = self
            .tool(tool)
            .and_then(|t| t.parameters.as_ref())
            .and_then(|p| p.get("properties"))
            .and_then(Value::as_object)
            .filter(|properties| !properties.is_empty());
        let repeated = given.contains(&key);
        match declared {
            Some(properties) if properties.contains_key(key) && !repeated => Some(Strength::Strong),
            None if !repeated && is_plausible_tool_name(key) => Some(Strength::Strong),
            _ if is_plausible_tool_name(key) => Some(Strength::Weak),
            _ => None,
        }
    }

    /// The name between `<tool_call>` and the tag after it, which must be `<arg_key>` or
    /// `</tool_call>`. `Err(true)` when undecided, `Err(false)` when there is none.
    fn call_name(&self, after_start: usize) -> Result<(&'a str, usize), bool> {
        match self.next_tag(after_start) {
            Some((at, Tag::KeyStart | Tag::CallEnd)) => {
                let name = self.text[after_start..at].trim();
                if name.is_empty() {
                    Err(false)
                } else {
                    Ok((name, at))
                }
            }
            Some(_) => Err(false),
            None => Err(!self.at_eof),
        }
    }

    /// Read the call whose `<tool_call>` starts at `start`.
    fn scan_call(&self, start: usize) -> Scan {
        let after_start = start + self.config.tool_call_start.len();
        let (name, mut pos) = match self.call_name(after_start) {
            Ok(found) => found,
            Err(true) => return Scan::Undecided,
            Err(false) => return Scan::Unparsed,
        };
        if self.name_strength(name).is_none() {
            return Scan::Unparsed;
        }
        let mut args: Vec<(String, std::ops::Range<usize>)> = Vec::new();
        loop {
            pos = self.skip_ws(pos);
            if pos == self.text.len() {
                return if self.at_eof {
                    Scan::Unparsed
                } else {
                    Scan::Undecided
                };
            }
            match (
                self.tag_at(pos, Tag::CallEnd),
                self.tag_at(pos, Tag::KeyStart),
            ) {
                (Some(true), _) => {
                    return Scan::Call(ScannedCall {
                        name: name.to_string(),
                        declared: self.tool(name).is_some(),
                        args,
                        end: pos + self.config.tool_call_end.len(),
                    });
                }
                (_, Some(true)) => {}
                (None, _) | (_, None) => return Scan::Undecided,
                _ => return Scan::Unparsed,
            }
            let (key, value_start) = match self.key_then_value(pos) {
                Ok(found) => found,
                Err(true) => return Scan::Undecided,
                Err(false) => return Scan::Unparsed,
            };
            let given: Vec<&str> = args.iter().map(|(k, _)| k.as_str()).collect();
            let given = [given.as_slice(), &[key]].concat();
            let value_end = match self.value_end(name, &given, value_start) {
                Ok(end) => end,
                Err(true) => return Scan::Undecided,
                Err(false) => return Scan::Unparsed,
            };
            args.push((key.to_string(), value_start..value_end));
            pos = value_end + self.config.arg_value_end.len();
        }
    }

    /// `<arg_key>KEY</arg_key>` at `pos`, optional whitespace, then `<arg_value>`: the key
    /// and where the value starts. `Err(true)` when undecided, `Err(false)` when malformed.
    fn key_then_value(&self, pos: usize) -> Result<(&'a str, usize), bool> {
        let key_start = pos + self.config.arg_key_start.len();
        let key_end = match self.next_tag(key_start) {
            Some((at, Tag::KeyEnd)) => at,
            Some(_) => return Err(false),
            None => return Err(!self.at_eof),
        };
        let key = self.text[key_start..key_end].trim();
        if key.is_empty() {
            return Err(false);
        }
        let value_tag = self.skip_ws(key_end + self.config.arg_key_end.len());
        if value_tag == self.text.len() {
            return Err(!self.at_eof);
        }
        match self.tag_at(value_tag, Tag::ValueStart) {
            Some(true) => Ok((key, value_tag + self.config.arg_value_start.len())),
            Some(false) => Err(false),
            None => Err(true),
        }
    }

    /// Where the value starting at `value_start` ends (the offset of its `</arg_value>`).
    /// `given` holds the keys of the call so far, this one included.
    fn value_end(&self, tool: &str, given: &[&str], value_start: usize) -> Result<usize, bool> {
        // First candidate of each class: strong and closed, weak and closed, strong, weak.
        let mut first: [Option<usize>; 4] = [None; 4];
        let mut balance = LiteralBalance::default();
        let mut pos = value_start;
        while let Some((at, tag)) = self.next_tag(pos) {
            if tag == Tag::ValueEnd {
                match self.continuation(tool, given, at) {
                    Continuation::Structural(strength) => {
                        let class = match (strength, balance.closed()) {
                            (Strength::Strong, true) => 0,
                            (Strength::Weak, true) => 1,
                            (Strength::Strong, false) => 2,
                            (Strength::Weak, false) => 3,
                        };
                        if class == 0 {
                            return Ok(at);
                        }
                        first[class].get_or_insert(at);
                    }
                    Continuation::Literal => {}
                    Continuation::Undecided => return Err(true),
                }
            }
            balance.add(tag);
            pos = at + tag.text(self.config).len();
        }
        if !self.at_eof {
            return Err(true);
        }
        first.into_iter().flatten().next().ok_or(false)
    }

    /// What follows the `</arg_value>` at `at`.
    fn continuation(&self, tool: &str, given: &[&str], at: usize) -> Continuation {
        let pos = self.skip_ws(at + self.config.arg_value_end.len());
        if pos == self.text.len() {
            // Closed value, no `</tool_call>`: a truncated call, not a reading to prefer.
            return if self.at_eof {
                Continuation::Literal
            } else {
                Continuation::Undecided
            };
        }
        match (
            self.tag_at(pos, Tag::KeyStart),
            self.tag_at(pos, Tag::CallEnd),
        ) {
            (Some(true), _) => match self.key_then_value(pos) {
                Ok((key, _)) => match self.key_strength(tool, key, given) {
                    Some(strength) => Continuation::Structural(strength),
                    None => Continuation::Literal,
                },
                Err(true) => Continuation::Undecided,
                Err(false) => Continuation::Literal,
            },
            (_, Some(true)) => self.after_call(pos + self.config.tool_call_end.len()),
            (None, _) | (_, None) => Continuation::Undecided,
            _ => Continuation::Literal,
        }
    }

    /// What follows a candidate `</tool_call>`.
    fn after_call(&self, pos: usize) -> Continuation {
        let pos = self.skip_ws(pos);
        if pos == self.text.len() {
            return if self.at_eof {
                Continuation::Structural(Strength::Strong)
            } else {
                Continuation::Undecided
            };
        }
        if self.tag_at(pos, Tag::CallStart).is_none() {
            return Continuation::Undecided;
        }
        // Optional text between calls, then either nothing more or the next call. Any
        // other tag after the text means the value goes on.
        match self.next_tag(pos) {
            Some((at, Tag::CallStart)) => {
                match self.call_name(at + self.config.tool_call_start.len()) {
                    Ok((name, _)) => match self.name_strength(name) {
                        Some(strength) => Continuation::Structural(strength),
                        None => Continuation::Literal,
                    },
                    // A later call cut off before its name ends: the earlier call is complete.
                    Err(_) if self.at_eof => Continuation::Structural(Strength::Strong),
                    Err(true) => Continuation::Undecided,
                    Err(false) => Continuation::Literal,
                }
            }
            Some(_) => Continuation::Literal,
            None if self.at_eof => Continuation::Structural(Strength::Strong),
            None => Continuation::Undecided,
        }
    }
}

impl Glm47Scanner<'_> {
    /// Whether the call ending at `end` is followed only by whitespace and then the end of
    /// the output or another `<tool_call>`. `None` while streaming and undecided.
    fn followed_by_call_or_end(&self, end: usize) -> Option<bool> {
        let pos = self.skip_ws(end);
        if pos == self.text.len() {
            return self.at_eof.then_some(true);
        }
        self.tag_at(pos, Tag::CallStart)
    }

    /// When the request declares tools, a call to an undeclared tool is the model's call
    /// only where calls stand: at the end of the output or right before another call.
    /// Followed by more text, it is markup the model wrote as text (a code block showing a
    /// tool-call fixture), not a call.
    fn stands_as_call(&self, call: &ScannedCall) -> Option<bool> {
        if call.declared || !self.declares_tools() {
            Some(true)
        } else {
            self.followed_by_call_or_end(call.end)
        }
    }
}

/// Where the GLM-4.7 calls at the start of a streamed buffer end, read with the request's
/// tool schemas (see [`Glm47Scanner`]).
#[derive(Debug, PartialEq, Eq)]
pub enum Glm47StreamBoundary {
    /// The calls before this byte offset are decided; the rest of the buffer is not part
    /// of them.
    Complete(usize),
    /// Keep buffering: a tag seen so far may still turn out to be text inside a value.
    Undecided,
    /// The buffer does not start with a call this reading applies to (bare call bodies,
    /// stray markup); use the plain end-marker handling.
    NotACall,
}

/// Find the decided end of the calls a streamed GLM-4.7 buffer starts with. A call is
/// decided once the text after it shows the next call; the last call of a response is
/// decided at the end of the stream, by the batch parser.
pub fn find_complete_tool_call_end_position_glm47(
    buffer: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> Glm47StreamBoundary {
    call_group_end(buffer, config, tools, false)
}

/// The end of the calls a complete GLM-4.7 output starts with (after any whitespace
/// between them), or `None` when it does not start with a call this reading applies to.
pub fn find_tool_call_group_end_glm47_at_eof(
    buffer: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> Option<usize> {
    match call_group_end(buffer, config, tools, true) {
        Glm47StreamBoundary::Complete(end) => Some(end),
        _ => None,
    }
}

fn call_group_end(
    buffer: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
    at_eof: bool,
) -> Glm47StreamBoundary {
    let scanner = Glm47Scanner::new(buffer, config, tools, at_eof);
    let mut pos = scanner.skip_ws(0);
    if !buffer[pos..].starts_with(config.tool_call_start.as_str()) {
        return Glm47StreamBoundary::NotACall;
    }
    let mut decided = None;
    loop {
        match scanner.scan_call(pos) {
            Scan::Call(call) if scanner.stands_as_call(&call) != Some(true) => break,
            Scan::Call(call) => {
                pos = scanner.skip_ws(call.end);
                if !buffer[pos..].starts_with(config.tool_call_start.as_str()) {
                    decided = Some(call.end);
                    break;
                }
                // Whitespace between two calls is framing, not content.
                decided = Some(pos);
            }
            Scan::Undecided => break,
            Scan::Unparsed if decided.is_none() => return Glm47StreamBoundary::NotACall,
            Scan::Unparsed => break,
        }
    }
    decided.map_or(
        Glm47StreamBoundary::Undecided,
        Glm47StreamBoundary::Complete,
    )
}

/// Whether `text` holds a GLM-4.7 tag, for callers deciding if a streamed buffer is worth
/// scanning again.
pub fn contains_glm47_tag(text: &str, config: &Glm47ParserConfig) -> bool {
    Tag::ALL
        .into_iter()
        .any(|tag| text.contains(tag.text(config)))
}

/// Build a call from its name and its `(key, raw value)` pairs: keys in the order the model
/// wrote them (a repeated key keeps its first position and its last value), values typed by
/// the parameter schema.
fn build_tool_call<'k>(
    function_name: String,
    pairs: impl IntoIterator<Item = (&'k str, &'k str)>,
    tools: Option<&[ToolDefinition]>,
) -> Result<ToolCallResponse, BlockError> {
    let mut arguments: Vec<(String, ParsedValue)> = Vec::new();
    let mut argument_indices: HashMap<&str, usize> = HashMap::new();
    for (key, raw_value) in pairs {
        if key.is_empty() {
            continue;
        }
        // The value is delivered as the model wrote it. GLM does not XML-escape
        // argument text, so `&amp;` in a value is source text (JSX, HTML), not an
        // escape: decoding it makes exact-match edit tools miss and rewrites the
        // code a write tool receives.
        let types = param_types(tools, &function_name, key);
        let json_value = coerce_param_value(raw_value, types);
        match argument_indices.get(key).copied() {
            Some(index) => arguments[index].1 = json_value,
            None => {
                argument_indices.insert(key, arguments.len());
                arguments.push((key.to_string(), json_value));
            }
        }
    }

    // A call to a tool the request does not declare is still the model's tool call:
    // return it with the name and arguments as written. Dropping it here left the
    // response with neither content nor tool_calls. Arguments of an undeclared tool have
    // no schema, so `coerce_value` keeps strings as written and parses only values that
    // are already JSON. An undeclared name must look like an identifier, though: markup
    // that lost its `<tool_call>` opener (a provider's broken output) otherwise yields a
    // "name" such as `1024</arg_value>`, and that block stays unparseable.
    let declared = tools.is_some_and(|tools| tools.iter().any(|t| t.name == function_name));
    if !declared && !is_plausible_tool_name(&function_name) {
        return Err(BlockError::UndeclaredNonIdentifier);
    }

    Ok(ToolCallResponse {
        id: Uuid::new_v4().to_string(),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: function_name,
            arguments: serde_json::to_string(&OrderedArguments(&arguments))
                .map_err(BlockError::Arguments)?,
        },
    })
}

/// Parse a single GLM-4.7 tool call block
/// Format: <tool_call>function_name<arg_key>key1</arg_key><arg_value>value1</arg_value>...</tool_call>
fn parse_tool_call_block(
    block: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> Result<ToolCallResponse, BlockError> {
    // Remove the outer <tool_call> tags
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    // Strip the outer start token. The end token is optional so we can
    // recover from max_tokens / EOS truncation that drops `</tool_call>`.
    let after_start = block
        .strip_prefix(start_token.as_str())
        .ok_or(BlockError::InvalidFormat)?;
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
        return Err(BlockError::EmptyFunctionName);
    }

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

    let regex = Regex::new(&pattern).map_err(BlockError::Pattern)?;
    let pairs: Vec<(&str, &str)> = regex
        .captures_iter(args_section)
        .map(|cap| {
            (
                cap.get(1).map(|m| m.as_str().trim()).unwrap_or(""),
                cap.get(2).map(|m| m.as_str()).unwrap_or(""),
            )
        })
        .collect();
    build_tool_call(function_name, pairs, tools)
}

/// Build the call [`Glm47Scanner`] read out of `text`.
fn build_scanned_call(
    text: &str,
    call: &ScannedCall,
    tools: Option<&[ToolDefinition]>,
) -> Result<ToolCallResponse, BlockError> {
    let pairs = call
        .args
        .iter()
        .map(|(key, value)| (key.as_str(), &text[value.clone()]));
    build_tool_call(call.name.clone(), pairs, tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_config() -> Glm47ParserConfig {
        Glm47ParserConfig::default()
    }

    fn edit_file_tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "edit_file".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "content": {"type": "string"},
                    "edits": {"type": "array"}
                }
            })),
            strict: None,
        }]
    }

    fn parse_args(input: &str) -> serde_json::Value {
        let (calls, _) =
            try_tool_call_parse_glm47(input, &get_test_config(), Some(&edit_file_tools())).unwrap();
        assert_eq!(calls.len(), 1);
        serde_json::from_str(&calls[0].function.arguments).unwrap()
    }

    #[test]
    fn test_argument_text_keeps_xml_entities() {
        let line =
            "<h2>System &amp; Intelligence &mdash; that&apos;s it &lt;3 &quot;x&quot; &gt;</h2>";
        let input = format!(
            "<tool_call>edit_file<arg_key>path</arg_key><arg_value>app.jsx</arg_value><arg_key>old_str</arg_key><arg_value>{line}</arg_value></tool_call>"
        );
        assert_eq!(parse_args(&input)["old_str"], line);
    }

    #[test]
    fn test_array_argument_keeps_xml_entities_inside_json() {
        let input = r#"<tool_call>edit_file<arg_key>path</arg_key><arg_value>app.jsx</arg_value><arg_key>edits</arg_key><arg_value>[{"old_str": "a &amp; b", "new_str": "a &amp; c"}]</arg_value></tool_call>"#;
        let args = parse_args(input);
        assert_eq!(args["edits"][0]["old_str"], "a &amp; b");
        assert_eq!(args["edits"][0]["new_str"], "a &amp; c");
    }

    #[test]
    fn test_array_argument_accepts_raw_control_characters_in_strings() {
        let input = "<tool_call>edit_file<arg_key>path</arg_key><arg_value>root.tsx</arg_value><arg_key>edits</arg_key><arg_value>[{\"old_str\": \"\t\t<CartProvider>\nnext, line\", \"new_str\": \"x\"}]</arg_value></tool_call>";
        let args = parse_args(input);
        assert_eq!(
            args["edits"][0]["old_str"],
            "\t\t<CartProvider>\nnext, line"
        );
    }

    #[test]
    fn test_broken_json_array_is_not_comma_split() {
        let input = r#"<tool_call>edit_file<arg_key>path</arg_key><arg_value>a.jsx</arg_value><arg_key>edits</arg_key><arg_value>[{"old_str": "f(a, b)" "old_str2": "x"}]</arg_value></tool_call>"#;
        let args = parse_args(input);
        assert!(
            args["edits"].is_string(),
            "broken JSON must reach the caller as written, got {}",
            args["edits"]
        );
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

        // A tool call block with no function name cannot be parsed — the whole
        // block (including <tool_call>...<arg_key>...<arg_value>... wire markup)
        // must be dropped, not leaked through normal_text.
        let message = "Here is the result: <tool_call><arg_key>x</arg_key><arg_value>1</arg_value></tool_call> done";
        let (calls, normal_text) =
            try_tool_call_parse_glm47(message, &config, Some(&tools)).unwrap();

        assert_eq!(calls.len(), 0);
        let text = normal_text.unwrap();
        assert!(
            !text.contains("<tool_call>") && !text.contains("<arg_key>"),
            "Wire-format tags must not leak into normal_text, got: {text}"
        );
        assert!(
            text.contains("Here is the result:") && text.contains("done"),
            "Surrounding prose must be preserved, got: {text}"
        );
    }

    fn image_tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "search".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"queries": {"type": "array"}}
            })),
            strict: None,
        }]
    }

    #[test]
    fn test_call_to_undeclared_tool_is_returned_as_written() {
        // The conversation history used a tool (img_gen) that the request no
        // longer declares, and the model calls it again.
        let message = "<tool_call>img_gen<arg_key>prompt</arg_key><arg_value>A \"bubbly\" logo, pastel pink</arg_value><arg_key>n</arg_key><arg_value>2</arg_value><arg_key>size</arg_key><arg_value>{\"w\": 1024}</arg_value></tool_call>";
        let (calls, normal_text) =
            try_tool_call_parse_glm47(message, &get_test_config(), Some(&image_tools())).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "img_gen");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"prompt":"A \"bubbly\" logo, pastel pink","n":"2","size":{"w":1024}}"#
        );
        assert_eq!(normal_text.as_deref(), Some(""));
    }

    #[test]
    fn test_undeclared_and_declared_calls_both_returned() {
        let message = "Generating now.<tool_call>img_gen<arg_key>prompt</arg_key><arg_value>a cat</arg_value></tool_call><tool_call>search<arg_key>queries</arg_key><arg_value>[\"cat logo\"]</arg_value></tool_call>";
        let (calls, normal_text) =
            try_tool_call_parse_glm47(message, &get_test_config(), Some(&image_tools())).unwrap();

        let names: Vec<_> = calls.iter().map(|c| c.function.name.as_str()).collect();
        assert_eq!(names, ["img_gen", "search"]);
        assert_eq!(calls[0].function.arguments, r#"{"prompt":"a cat"}"#);
        assert_eq!(calls[1].function.arguments, r#"{"queries":["cat logo"]}"#);
        assert_eq!(normal_text.as_deref(), Some("Generating now."));
    }

    #[test]
    fn test_undeclared_name_must_be_an_identifier() {
        assert!(is_plausible_tool_name("img_gen"));
        assert!(is_plausible_tool_name("functions.search:v2-beta"));
        assert!(is_plausible_tool_name("_private"));
        assert!(!is_plausible_tool_name("1024</arg_value>"));
        assert!(!is_plausible_tool_name("logo</arg_value>"));
        assert!(!is_plausible_tool_name("2fast"));
        assert!(!is_plausible_tool_name("a red logo"));
        assert!(!is_plausible_tool_name(""));
        assert!(!is_plausible_tool_name(&"a".repeat(129)));
    }

    #[test]
    fn test_garbage_named_block_is_dropped() {
        // A provider returned the call's tail as content without `<tool_call>img_gen`:
        // the bare-body recovery takes the word before the first marker as the name.
        // Either the parse fails (the jail then releases no content) or it yields no call.
        let no_call = |message: &str| {
            try_tool_call_parse_glm47(message, &get_test_config(), Some(&image_tools())).map_or(
                true,
                |(calls, text)| {
                    calls.is_empty() && !text.unwrap_or_default().contains("</arg_value>")
                },
            )
        };
        assert!(no_call(
            "</arg_key><arg_value>1024</arg_value><arg_key>prompt</arg_key><arg_value>A dark logo</arg_value></tool_call>"
        ));
        assert!(
            no_call(
                "a minimalist logo</arg_value><arg_key>width</arg_key><arg_value>1024</arg_value></tool_call>"
            ),
            "`logo</arg_value>` must not become a tool call"
        );
        // Inside a complete block the garbage name is dropped like any unparseable block.
        let message = "<tool_call>768</arg_value><arg_key>prompt</arg_key><arg_value>x</arg_value></tool_call>";
        let (calls, normal_text) =
            try_tool_call_parse_glm47(message, &get_test_config(), Some(&image_tools())).unwrap();
        assert!(calls.is_empty(), "got {calls:?}");
        assert_eq!(normal_text.as_deref(), Some(""));
        // Without declared tools a garbage name is not a call either.
        let message =
            "<tool_call>logo</arg_value><arg_key>x</arg_key><arg_value>1</arg_value></tool_call>";
        let (calls, _) = try_tool_call_parse_glm47(message, &get_test_config(), None).unwrap();
        assert!(calls.is_empty(), "got {calls:?}");
    }

    #[test]
    fn test_declared_name_is_not_checked_as_an_identifier() {
        let tools = vec![ToolDefinition {
            name: "lookup weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let message = "<tool_call>lookup weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
        let (calls, _) =
            try_tool_call_parse_glm47(message, &get_test_config(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup weather");
    }

    #[test]
    fn test_bare_body_call_to_undeclared_tool_is_recovered() {
        // `<tool_call>` itself missing (bare-body recovery path): an undeclared
        // name used to make the whole parse fail.
        let message = "img_gen<arg_key>prompt</arg_key><arg_value>a cat</arg_value></tool_call>";
        let (calls, normal_text) =
            try_tool_call_parse_glm47(message, &get_test_config(), Some(&image_tools())).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "img_gen");
        assert_eq!(calls[0].function.arguments, r#"{"prompt":"a cat"}"#);
        assert_eq!(normal_text.as_deref(), Some(""));
    }

    #[test] // helper
    fn test_xml_entities_are_not_decoded() {
        let config = get_test_config();
        let message = r#"<tool_call>write_file<arg_key>content</arg_key><arg_value>x &lt; y &amp;&amp; y &gt; z</arg_value></tool_call>"#;

        let (calls, _) = try_tool_call_parse_glm47(message, &config, None).unwrap();

        assert_eq!(calls.len(), 1);
        let args: HashMap<String, Value> =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args.get("content").unwrap().as_str().unwrap(),
            "x &lt; y &amp;&amp; y &gt; z"
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

    // ---------------------------------------------------------------------------------
    // GLM markup inside argument values. GLM writes string arguments raw, so a value can
    // hold `<tool_call>`, `</arg_value>`, `</tool_call>` as text.
    // ---------------------------------------------------------------------------------

    fn bash_tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "bash".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            })),
            strict: None,
        }]
    }

    fn edit_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "edit".to_string(),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"}
                    }
                })),
                strict: None,
            },
            bash_tools().remove(0),
        ]
    }

    fn bash_call(command: &str) -> String {
        format!(
            "<tool_call>bash<arg_key>command</arg_key><arg_value>{command}</arg_value></tool_call>"
        )
    }

    /// (name, arguments) of each call, and the normal text.
    fn parse_with(message: &str, tools: &[ToolDefinition]) -> (Vec<(String, Value)>, String) {
        let (calls, text) =
            try_tool_call_parse_glm47(message, &get_test_config(), Some(tools)).unwrap();
        let calls = calls
            .into_iter()
            .map(|c| {
                (
                    c.function.name,
                    serde_json::from_str(&c.function.arguments).unwrap(),
                )
            })
            .collect();
        (calls, text.unwrap_or_default())
    }

    const FIXTURE: &str =
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";

    #[test]
    fn test_markup_fixture_in_heredoc_stays_in_the_command() {
        for command in [
            format!("cat > /tmp/fixture.txt <<'EOF'\n{FIXTURE}\nEOF"),
            format!("cat > /tmp/fixture.txt <<'EOF'\n{FIXTURE}\n{FIXTURE}\nEOF\ncat /tmp/fixture.txt"),
            // GLM-4.5 layout, newlines between the elements.
            "cat > f <<'EOF'\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>\nEOF".to_string(),
            // The fixture is a call to a declared tool.
            format!("cat > f <<'EOF'\n{}\nEOF", bash_call("ls")),
            // The command ends with the fixture.
            format!("printf '%s' '{FIXTURE}'"),
            FIXTURE.to_string(),
        ] {
            let (calls, text) = parse_with(&bash_call(&command), &bash_tools());
            assert_eq!(calls, [("bash".to_string(), serde_json::json!({"command": command}))]);
            assert_eq!(text, "");
        }
    }

    #[test]
    fn test_other_formats_markup_stays_in_the_command() {
        // Qwen / Hermes markup shares the tag names: `<tool_call>` with no `</arg_value>`.
        for command in [
            "python3 -c 'print(\"<tool_call>\\n{\\\"name\\\": \\\"f\\\"}\\n</tool_call>\")'",
            "grep -n '</tool_call>' parser.py",
            "grep -n '<tool_call>' parser.py && sed -i 's#</arg_value>##' x.txt",
            "echo '<arg_key>city</arg_key> value: <arg_value>Paris</arg_value>'",
            "echo 'key: <tool_call>city</arg_key> value: <arg_key>Paris'",
            "if [[ $s == *'</arg_value></tool_call>'* ]]; then echo closed; fi",
        ] {
            let (calls, text) = parse_with(&bash_call(command), &bash_tools());
            assert_eq!(
                calls,
                [("bash".to_string(), serde_json::json!({"command": command}))],
                "{command}"
            );
            assert_eq!(text, "");
        }
    }

    #[test]
    fn test_recorded_outputs_with_markup_in_the_command() {
        // GLM-5.3 and GLM-5.3-Flash outputs for Terminal-Bench parser tasks. The token ids
        // show the model wrote the markup inside the command as ordinary text and only the
        // outer tags as tag tokens, so the command is the text between the first
        // `<arg_value>` and the last `</arg_value>`. Before this reading every one parsed
        // as `bash({})` with the rest of the command in content.
        for (name, text) in [
            (
                "qwen_parser_test_script",
                include_str!(
                    "../../../tests/data/glm47_markup_in_values/glm53_qwen_parser_test_script.txt"
                ),
            ),
            (
                "qwen_parser_test_script_2",
                include_str!(
                    "../../../tests/data/glm47_markup_in_values/glm53_qwen_parser_test_script_2.txt"
                ),
            ),
            (
                "flash_parser_samples_script",
                include_str!(
                    "../../../tests/data/glm47_markup_in_values/glm53_flash_parser_samples_script.txt"
                ),
            ),
            (
                "flash_fixture_heredoc",
                include_str!(
                    "../../../tests/data/glm47_markup_in_values/glm53_flash_fixture_heredoc.txt"
                ),
            ),
        ] {
            let start = text.find("<arg_value>").unwrap() + "<arg_value>".len();
            let end = text.rfind("</arg_value>").unwrap();
            let prose = &text[..text.find("<tool_call>").unwrap()];
            let (calls, normal) = parse_with(text, &bash_tools());
            assert_eq!(
                calls,
                [(
                    "bash".to_string(),
                    serde_json::json!({"command": &text[start..end]})
                )],
                "{name}"
            );
            assert_eq!(normal, prose, "{name}");
        }
    }

    #[test]
    fn test_parallel_calls_with_markup_in_values() {
        let first = format!("cat > a.txt <<'EOF'\n{FIXTURE}\nEOF");
        let message = format!(
            "{}\n{}",
            bash_call(&first),
            bash_call("grep -c '</tool_call>' a.txt")
        );
        let (calls, _) = parse_with(&message, &bash_tools());
        assert_eq!(
            calls,
            [
                ("bash".to_string(), serde_json::json!({"command": first})),
                (
                    "bash".to_string(),
                    serde_json::json!({"command": "grep -c '</tool_call>' a.txt"})
                ),
            ]
        );

        // A value with an unclosed literal opener, then a second call.
        let message = format!(
            "{}{}",
            bash_call("grep -n '<tool_call>' p.py"),
            bash_call("ls")
        );
        let (calls, _) = parse_with(&message, &bash_tools());
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert_eq!(calls[0].1["command"], "grep -n '<tool_call>' p.py");
        assert_eq!(calls[1].1["command"], "ls");

        // The second call is to an undeclared tool: still two calls.
        let message = format!(
            "{}<tool_call>read_file<arg_key>path</arg_key><arg_value>a.txt</arg_value></tool_call>",
            bash_call("ls")
        );
        let (calls, _) = parse_with(&message, &bash_tools());
        assert_eq!(
            calls[0],
            ("bash".to_string(), serde_json::json!({"command": "ls"}))
        );
        assert_eq!(
            calls[1],
            (
                "read_file".to_string(),
                serde_json::json!({"path": "a.txt"})
            )
        );
    }

    #[test]
    fn test_multi_parameter_tool_with_markup_in_values() {
        let old_str = "x = \"</arg_value>\"\nend = \"</tool_call>\"";
        let new_str = format!("FIXTURE = \"{FIXTURE}\"");
        let message = format!(
            "<tool_call>edit<arg_key>path</arg_key><arg_value>t.py</arg_value><arg_key>old_str</arg_key><arg_value>{old_str}</arg_value><arg_key>new_str</arg_key><arg_value>{new_str}</arg_value></tool_call>"
        );
        let (calls, text) = parse_with(&message, &edit_tools());
        assert_eq!(
            calls,
            [(
                "edit".to_string(),
                serde_json::json!({"path": "t.py", "old_str": old_str, "new_str": new_str})
            )]
        );
        assert_eq!(text, "");

        // A value holding a complete `<arg_key>path</arg_key><arg_value>` sequence: `path`
        // is already given and the literal pairs only close at the real end, so it stays
        // text in the value.
        let new_str = "a <arg_value>1</arg_value><arg_key>path</arg_key><arg_value>2</arg_value> b";
        let message = format!(
            "<tool_call>edit<arg_key>path</arg_key><arg_value>t.py</arg_value><arg_key>new_str</arg_key><arg_value>{new_str}</arg_value></tool_call>"
        );
        let (calls, _) = parse_with(&message, &edit_tools());
        assert_eq!(
            calls[0].1,
            serde_json::json!({"path": "t.py", "new_str": new_str})
        );
    }

    #[test]
    fn test_undeclared_parameters_and_repeated_keys_stay_parameters() {
        // No reading with declared keys closes every literal pair, so the undeclared and
        // repeated keys are the model's arguments, as before.
        let message = "<tool_call>edit<arg_key>path</arg_key><arg_value>a.py</arg_value><arg_key>old_string</arg_key><arg_value>x</arg_value><arg_key>new_string</arg_key><arg_value>y</arg_value></tool_call>";
        let (calls, _) = parse_with(message, &edit_tools());
        assert_eq!(
            calls[0].1,
            serde_json::json!({"path": "a.py", "old_string": "x", "new_string": "y"})
        );
        let message = "<tool_call>edit<arg_key>path</arg_key><arg_value>a.py</arg_value><arg_key>path</arg_key><arg_value>b.py</arg_value></tool_call>";
        let (calls, _) = parse_with(message, &edit_tools());
        assert_eq!(calls[0].1, serde_json::json!({"path": "b.py"}));
    }

    #[test]
    fn test_scanned_end_matches_batch_parse_while_streaming() {
        // The streamed boundary decides a call only once the next call shows, and never
        // inside a value.
        let config = get_test_config();
        let tools = bash_tools();
        let first = bash_call(&format!("cat > a <<'EOF'\n{FIXTURE}\nEOF"));
        let message = format!("{first}{}", bash_call("ls"));
        for cut in 0..=message.len() {
            if !message.is_char_boundary(cut) {
                continue;
            }
            let boundary =
                find_complete_tool_call_end_position_glm47(&message[..cut], &config, Some(&tools));
            match boundary {
                Glm47StreamBoundary::Complete(end) => {
                    assert_eq!(end, first.len(), "cut {cut}");
                    assert!(cut >= first.len() + "<tool_call>bash<".len(), "cut {cut}");
                }
                Glm47StreamBoundary::Undecided => {
                    assert!(
                        cut < first.len() + "<tool_call>bash<arg_key>".len(),
                        "cut {cut}"
                    )
                }
                Glm47StreamBoundary::NotACall => assert!(cut < "<tool_call>".len(), "cut {cut}"),
            }
        }
    }

    #[test]
    fn test_undeclared_call_followed_by_text_is_text() {
        // GLM-5.3-Flash showed the fixture in a code block, then made the call. The code
        // block is ordinary text (the tag tokens appear only in the call): a call to an
        // undeclared tool counts only where calls stand, at the end or before another call.
        let block = format!("```bash\ncat > /tmp/fixture.txt <<'EOF'\n{FIXTURE}\nEOF\n```");
        for (message, prose) in [
            (
                format!(
                    "{block}\n\nTo verify it:{}",
                    bash_call("cat /tmp/fixture.txt")
                ),
                format!("{block}\n\nTo verify it:"),
            ),
            (
                format!("{block}{}", bash_call("cat /tmp/fixture.txt")),
                block.clone(),
            ),
        ] {
            let (calls, text) = parse_with(&message, &bash_tools());
            assert_eq!(
                calls,
                [(
                    "bash".to_string(),
                    serde_json::json!({"command": "cat /tmp/fixture.txt"})
                )]
            );
            assert_eq!(text, prose);
        }
        // A call to an undeclared tool at the end of the output is still returned.
        let (calls, _) = parse_with(&format!("Calling it.{FIXTURE}"), &bash_tools());
        assert_eq!(
            calls,
            [(
                "get_weather".to_string(),
                serde_json::json!({"city": "Paris"})
            )]
        );
    }

    #[test]
    fn test_provider_value_tail_is_not_a_bare_call() {
        // A provider lost `<tool_call>img_gen<arg_key>prompt</arg_key><arg_value>` and
        // returned the rest as content. A call name is followed by `<arg_key>` or
        // `</tool_call>`, so `logo</arg_value>…` is the tail of a value, not a call body.
        for message in [
            "a minimalist logo</arg_value><arg_key>width</arg_key><arg_value>1024</arg_value></tool_call>",
            "flat 2D illustration style</arg_value><arg_key>width</arg_key><arg_value>1024</arg_value><arg_key>height</arg_key><arg_value>768</arg_value></tool_call>",
        ] {
            let (calls, text) =
                try_tool_call_parse_glm47(message, &get_test_config(), Some(&image_tools()))
                    .unwrap();
            assert!(calls.is_empty(), "{calls:?}");
            assert!(!text.unwrap_or_default().contains("</arg_value>"));
            assert_eq!(
                find_tool_call_end_position_glm47(message, &get_test_config()),
                message.len()
            );
        }
    }
}

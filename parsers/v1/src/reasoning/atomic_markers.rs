// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reasoning markers that are single special tokens.
//!
//! Several model families write their reasoning and tool-call markers (`<think>`, `</think>`,
//! `<tool_call>`) as one added token each. The same characters can also appear as ordinary text:
//! a model that quotes a prompt asking for `<think></think>` tags writes them as several ordinary
//! tokens inside its reasoning. A text-only parser cannot tell the two apart, and treats the quoted
//! `</think>` as the end of the reasoning block, so the rest of the thinking reaches the client as
//! content.
//!
//! [`AtomicMarkerParser`] uses the chunk's token ids. A marker that is a single token arrives whole
//! within one chunk, so its text in a chunk whose ids do not include that token is ordinary text,
//! and so is a marker prefix at the end of a chunk. Those occurrences are hidden from the inner
//! parser (their leading `<` is swapped for a private sentinel) and restored in its output. Chunks
//! without token ids are passed through unchanged.

use super::{ParserResult, ReasoningParser};

/// Stands in for the `<` of a marker occurrence that is ordinary text. Two private-use characters,
/// restored to `<` in every result.
const SENTINEL: &str = "\u{E000}\u{E001}";

#[derive(Debug)]
pub struct AtomicMarkerParser {
    inner: Box<dyn ReasoningParser>,
    /// Marker text and its token id, for markers that are one token in the model's tokenizer.
    markers: Vec<(String, u32)>,
}

impl AtomicMarkerParser {
    pub fn new(inner: Box<dyn ReasoningParser>, markers: Vec<(String, u32)>) -> Self {
        let markers = markers
            .into_iter()
            .filter(|(text, _)| text.starts_with('<') && text.len() > 1)
            .collect();
        Self { inner, markers }
    }

    /// The chunk text with every marker occurrence that is ordinary text neutralized.
    fn escape(&self, text: &str, token_ids: &[u32]) -> String {
        if self.markers.is_empty() || !text.contains('<') {
            return text.to_string();
        }
        // Byte offsets of the `<` to neutralize.
        let mut literal = Vec::new();
        for (marker, id) in &self.markers {
            let special = token_ids.iter().filter(|t| *t == id).count();
            let found: Vec<usize> = text
                .match_indices(marker.as_str())
                .map(|(at, _)| at)
                .collect();
            // Occurrences beyond the number of special tokens in this chunk are ordinary text. Within
            // one chunk the order is unknown; the first ones are taken as the special tokens.
            literal.extend(found.into_iter().skip(special));
        }
        // A marker prefix at the end of the chunk cannot be the start of a special token, which
        // would have arrived whole.
        if let Some(lt) = text.rfind('<') {
            let tail = &text[lt..];
            let is_proper_prefix = self
                .markers
                .iter()
                .any(|(marker, _)| tail.len() < marker.len() && marker.starts_with(tail));
            if is_proper_prefix {
                literal.push(lt);
            }
        }
        if literal.is_empty() {
            return text.to_string();
        }
        literal.sort_unstable();
        literal.dedup();
        let mut out = String::with_capacity(text.len() + literal.len() * SENTINEL.len());
        let mut last = 0;
        for at in literal {
            out.push_str(&text[last..at]);
            out.push_str(SENTINEL);
            last = at + 1;
        }
        out.push_str(&text[last..]);
        out
    }

    fn restore(mut result: ParserResult) -> ParserResult {
        if result.normal_text.contains(SENTINEL) {
            result.normal_text = result.normal_text.replace(SENTINEL, "<");
        }
        if result.reasoning_text.contains(SENTINEL) {
            result.reasoning_text = result.reasoning_text.replace(SENTINEL, "<");
        }
        result
    }
}

impl ReasoningParser for AtomicMarkerParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        // One-shot input carries the whole output's ids, which do not say which occurrence is which.
        self.inner.detect_and_parse_reasoning(text, token_ids)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        if token_ids.is_empty() {
            return self
                .inner
                .parse_reasoning_streaming_incremental(text, token_ids);
        }
        let escaped = self.escape(text, token_ids);
        Self::restore(
            self.inner
                .parse_reasoning_streaming_incremental(&escaped, token_ids),
        )
    }

    fn finish_reasoning_stream(&mut self) -> ParserResult {
        Self::restore(self.inner.finish_reasoning_stream())
    }

    fn set_in_reasoning(&mut self, in_reasoning: bool) {
        self.inner.set_in_reasoning(in_reasoning)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning::ReasoningParserType;

    const THINK: u32 = 154841;
    const THINK_END: u32 = 154842;
    const TOOL_CALL: u32 = 154843;

    fn glm45() -> AtomicMarkerParser {
        let mut inner = Box::new(ReasoningParserType::get_reasoning_parser_from_name("glm45"))
            as Box<dyn ReasoningParser>;
        inner.set_in_reasoning(true);
        AtomicMarkerParser::new(
            inner,
            vec![
                ("<think>".into(), THINK),
                ("</think>".into(), THINK_END),
                ("<tool_call>".into(), TOOL_CALL),
            ],
        )
    }

    fn run(parser: &mut AtomicMarkerParser, chunks: &[(&str, &[u32])]) -> (String, String) {
        let (mut reasoning, mut normal) = (String::new(), String::new());
        for (text, ids) in chunks {
            let r = parser.parse_reasoning_streaming_incremental(text, ids);
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        let r = parser.finish_reasoning_stream();
        reasoning.push_str(&r.reasoning_text);
        normal.push_str(&r.normal_text);
        (reasoning, normal)
    }

    /// Recorded GLM-5.3 completion shape (public K2VV request): the prompt asked for `<think>`
    /// tags, the model quoted them inside its reasoning as ordinary tokens, then closed the block
    /// with the special token.
    #[test]
    fn quoted_think_tags_in_ordinary_tokens_stay_reasoning() {
        let mut parser = glm45();
        let (reasoning, normal) = run(
            &mut parser,
            &[
                ("should be enclosed in", &[43560, 304]),
                (
                    " <think></think> tags.",
                    &[366, 26779, 1472, 26779, 29, 9488, 13],
                ),
                (" Done.", &[2]),
                ("</think>", &[THINK_END]),
                ("The answer.", &[785, 4226]),
            ],
        );
        assert_eq!(
            reasoning,
            "should be enclosed in <think></think> tags. Done."
        );
        assert_eq!(normal, "The answer.");
    }

    #[test]
    fn quoted_close_split_across_chunks_stays_reasoning() {
        let mut parser = glm45();
        let (reasoning, normal) = run(
            &mut parser,
            &[
                ("tags like </", &[9488, 1075, 522]),
                ("think> here.", &[26779, 1339, 1588]),
                ("</think>", &[THINK_END]),
                ("Answer", &[16141]),
            ],
        );
        assert_eq!(reasoning, "tags like </think> here.");
        assert_eq!(normal, "Answer");
    }

    #[test]
    fn special_close_ends_reasoning() {
        let mut parser = glm45();
        let (reasoning, normal) = run(
            &mut parser,
            &[
                ("Thinking.", &[1, 2]),
                (" Done.</think>Hi", &[3, THINK_END, 4]),
            ],
        );
        assert_eq!(reasoning, "Thinking. Done.");
        assert_eq!(normal, "Hi");
    }

    #[test]
    fn special_tool_call_still_ends_reasoning_at_end_of_stream() {
        let mut parser = glm45();
        let (reasoning, normal) = run(
            &mut parser,
            &[
                ("Let me search.", &[1, 2, 3]),
                ("<tool_call>search", &[TOOL_CALL, 5]),
                (
                    "<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>",
                    &[6, 7, 8, 9, 10, 11, 12],
                ),
            ],
        );
        assert_eq!(reasoning, "Let me search.");
        assert_eq!(
            normal,
            "<tool_call>search<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>"
        );
    }

    #[test]
    fn chunks_without_token_ids_parse_as_before() {
        let text = "a <think></think> b</think>c";
        let mut plain = ReasoningParserType::get_reasoning_parser_from_name("glm45");
        plain.set_in_reasoning(true);
        let expected = plain.parse_reasoning_streaming_incremental(text, &[]);
        let got = glm45().parse_reasoning_streaming_incremental(text, &[]);
        assert_eq!(got.reasoning_text, expected.reasoning_text);
        assert_eq!(got.normal_text, expected.normal_text);
    }

    #[test]
    fn escape_restores_every_character() {
        let parser = glm45();
        let text = "x </think> y <tool_call> z </th";
        let escaped = parser.escape(text, &[1, 2, 3]);
        assert!(!escaped.contains("</think>") && !escaped.contains("<tool_call>"));
        assert_eq!(escaped.replace(SENTINEL, "<"), text);
    }
}

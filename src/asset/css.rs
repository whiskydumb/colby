//! A deliberate subset of CSS.
//!
//! Not a browser. The subset is what an interface actually reaches for:
//! selectors are one tag, one class, one identifier and an optional `:hover`,
//! with no combinators; properties are the flexbox set plus the handful of
//! paint properties a box needs. A stylesheet reader of that shape is a couple
//! of hundred lines, and that is the size this problem is when nobody tries to
//! be a browser.
//!
//! **Nothing here is an error.** A property nobody implemented, a selector with
//! two classes in it, a color that is not a color - every one of them is a
//! warning carrying the text that caused it, collected and handed back. The
//! compiler logs them with the file's name, once, at compile time. The
//! alternative is the thing that costs an hour: a stylesheet where one
//! misspelled property silently does nothing and everything around it works.
//!
//! What is deliberately missing, and why:
//!
//! - **combinators and descendant selectors.** A flat set of classes styles an
//!   interface this size, and matching without them is four string compares.
//! - **`text-align`.** A text node's box is exactly its text, so centering it
//!   is `justify-content: center` on the box around it. The property would be a
//!   second way to say the same thing.
//! - **`em`, `rem`, `vw`, `vh`.** Nothing has needed one. Each is a line here
//!   and an arm in the layout on the day something does.
//! - **`border`, `box-shadow`, `transition`.** Each is real work in the
//!   renderer rather than in the parser, and the last also needs per-node state
//!   that survives a step.

use colby_core::abi::ui::{
	Rule, Selector,
	style::{Align, Color, Direction, Display, Edges, Justify, Length, Position, Style, Wrap},
};

/// What a stylesheet turned into.
#[derive(Clone, Debug, Default)]
pub struct Sheet {
	/// Every rule, in the order they were written.
	pub rules: Vec<Rule>,

	/// Everything the parser did not understand, in the words it was given.
	pub warnings: Vec<String>,
}

/// Reads a stylesheet.
///
/// @param text - the whole sheet
/// @return the rules, and whatever could not be read
#[must_use]
pub fn parse(text: &str) -> Sheet {
	let mut sheet = Sheet::default();
	let stripped = strip_comments(text);
	let mut rest = stripped.as_str();

	while let Some(open) = rest.find('{') {
		let (selectors, tail) = rest.split_at(open);
		let Some(close) = tail.find('}') else {
			sheet
				.warnings
				.push(format!("a rule for `{}` is never closed", selectors.trim()));

			break;
		};

		let body = tail.get(1..close).unwrap_or_default().to_owned();
		rest = tail.get(close + 1..).unwrap_or_default();

		let style = declarations(&body, &mut sheet.warnings);

		for selector in selectors.split(',') {
			let trimmed = selector.trim();
			if trimmed.is_empty() {
				continue;
			}

			match compound(trimmed) {
				| Ok(selector) => sheet
					.rules
					.push(Rule { selector, style: style.clone() }),
				| Err(reason) => sheet.warnings.push(reason),
			}
		}
	}

	sheet
}

/// Reads the body of a `style` attribute, or the inside of one rule.
///
/// @param text - `property: value; property: value`
/// @param warnings - where anything unreadable is recorded
/// @return everything that was understood
pub fn declarations(text: &str, warnings: &mut Vec<String>) -> Style {
	let mut style = Style::default();

	for declaration in text.split(';') {
		let trimmed = declaration.trim();
		if trimmed.is_empty() {
			continue;
		}

		let Some((property, value)) = trimmed.split_once(':') else {
			warnings.push(format!("`{trimmed}` is not a `property: value` pair"));

			continue;
		};

		let property = property.trim().to_ascii_lowercase();
		let value = value.trim();

		if !apply(&mut style, &property, value) {
			warnings.push(format!("`{property}: {value}` is not something colby draws"));
		}
	}

	style
}

/// Sets one property, if it is one this build knows.
///
/// @return whether anything was written
fn apply(style: &mut Style, property: &str, value: &str) -> bool {
	box_property(style, property, value)
		|| flex_property(style, property, value)
		|| paint_property(style, property, value)
		|| edge_property(style, property, value)
}

/// Size, position and the other properties about the box itself.
fn box_property(style: &mut Style, property: &str, value: &str) -> bool {
	match property {
		| "display" => match value {
			| "flex" => style.display = Some(Display::Flex),
			| "none" => style.display = Some(Display::None),
			| _ => return false,
		},
		| "position" => match value {
			| "relative" => style.position = Some(Position::Relative),
			| "absolute" => style.position = Some(Position::Absolute),
			| _ => return false,
		},
		| "width" => style.width = length(value),
		| "height" => style.height = length(value),
		| "min-width" => style.min_width = length(value),
		| "min-height" => style.min_height = length(value),
		| "max-width" => style.max_width = length(value),
		| "max-height" => style.max_height = length(value),
		| _ => return false,
	}

	// a property that is known but whose value is not is still a failure: the
	// match above wrote `None` into it, and reporting that as understood is how
	// `width: 12rem` turns into a box that is silently the wrong size.
	written(style, property)
}

/// The flexbox properties.
fn flex_property(style: &mut Style, property: &str, value: &str) -> bool {
	match property {
		| "flex-direction" => match value {
			| "row" => style.direction = Some(Direction::Row),
			| "row-reverse" => style.direction = Some(Direction::RowReverse),
			| "column" => style.direction = Some(Direction::Column),
			| "column-reverse" => style.direction = Some(Direction::ColumnReverse),
			| _ => return false,
		},
		| "flex-wrap" => match value {
			| "nowrap" => style.wrap = Some(Wrap::NoWrap),
			| "wrap" => style.wrap = Some(Wrap::Wrap),
			| _ => return false,
		},
		| "justify-content" => match value {
			| "flex-start" | "start" => style.justify = Some(Justify::Start),
			| "flex-end" | "end" => style.justify = Some(Justify::End),
			| "center" => style.justify = Some(Justify::Center),
			| "space-between" => style.justify = Some(Justify::SpaceBetween),
			| "space-around" => style.justify = Some(Justify::SpaceAround),
			| "space-evenly" => style.justify = Some(Justify::SpaceEvenly),
			| _ => return false,
		},
		| "align-items" => match value {
			| "flex-start" | "start" => style.align = Some(Align::Start),
			| "flex-end" | "end" => style.align = Some(Align::End),
			| "center" => style.align = Some(Align::Center),
			| "stretch" => style.align = Some(Align::Stretch),
			| _ => return false,
		},
		| "flex-grow" => style.grow = number(value),
		| "flex-shrink" => style.shrink = number(value),
		| "flex-basis" => style.basis = length(value),
		| "gap" => style.gap = length(value),
		| _ => return false,
	}

	written(style, property)
}

/// What a box is painted with.
fn paint_property(style: &mut Style, property: &str, value: &str) -> bool {
	match property {
		| "background-color" | "background" => style.background = color(value),
		| "color" => style.color = color(value),
		| "font-size" => style.font_size = length(value),
		| "font-family" => style.font_family = Some(unquote(value).to_owned()),
		| "border-radius" => style.radius = length(value),
		| "opacity" => style.opacity = number(value),
		| _ => return false,
	}

	written(style, property)
}

/// Margin, padding, and where an absolutely positioned box sits.
fn edge_property(style: &mut Style, property: &str, value: &str) -> bool {
	if let Some(side) = property.strip_prefix("margin") {
		return edge(&mut style.margin, side, value);
	}

	if let Some(side) = property.strip_prefix("padding") {
		return edge(&mut style.padding, side, value);
	}

	match property {
		| "left" => style.inset.left = length(value),
		| "right" => style.inset.right = length(value),
		| "top" => style.inset.top = length(value),
		| "bottom" => style.inset.bottom = length(value),
		| "inset" => return shorthand(&mut style.inset, value),
		| _ => return false,
	}

	!style.inset.is_unset()
}

/// One side of an edge property, or all four of them.
///
/// @param edges - what to write into
/// @param side - what followed `margin` or `padding`: `-left`, or nothing
/// @param value - the length, or up to four of them
fn edge(edges: &mut Edges, side: &str, value: &str) -> bool {
	let Some(length) = length(value) else {
		return side.is_empty() && shorthand(edges, value);
	};

	match side {
		| "" => *edges = Edges::all(length),
		| "-left" => edges.left = Some(length),
		| "-right" => edges.right = Some(length),
		| "-top" => edges.top = Some(length),
		| "-bottom" => edges.bottom = Some(length),
		| _ => return false,
	}

	true
}

/// The one-to-four-value form: `8px`, `8px 12px`, or all four, clockwise.
fn shorthand(edges: &mut Edges, value: &str) -> bool {
	let parts: Vec<Length> = value
		.split_whitespace()
		.filter_map(length)
		.collect();

	match parts.as_slice() {
		| [all] => *edges = Edges::all(*all),
		| [vertical, horizontal] =>
			*edges = Edges {
				top: Some(*vertical),
				bottom: Some(*vertical),
				left: Some(*horizontal),
				right: Some(*horizontal),
			},
		| [top, horizontal, bottom] =>
			*edges = Edges {
				top: Some(*top),
				left: Some(*horizontal),
				right: Some(*horizontal),
				bottom: Some(*bottom),
			},
		| [top, right, bottom, left] =>
			*edges = Edges {
				top: Some(*top),
				right: Some(*right),
				bottom: Some(*bottom),
				left: Some(*left),
			},
		| _ => return false,
	}

	true
}

/// Whether the property just handled ended up holding a value.
///
/// The properties whose value is parsed rather than matched write `None` when
/// they cannot read it, and this is what turns that back into "not understood".
fn written(style: &Style, property: &str) -> bool {
	match property {
		| "width" => style.width.is_some(),
		| "height" => style.height.is_some(),
		| "min-width" => style.min_width.is_some(),
		| "min-height" => style.min_height.is_some(),
		| "max-width" => style.max_width.is_some(),
		| "max-height" => style.max_height.is_some(),
		| "flex-grow" => style.grow.is_some(),
		| "flex-shrink" => style.shrink.is_some(),
		| "flex-basis" => style.basis.is_some(),
		| "gap" => style.gap.is_some(),
		| "background-color" | "background" => style.background.is_some(),
		| "color" => style.color.is_some(),
		| "font-size" => style.font_size.is_some(),
		| "border-radius" => style.radius.is_some(),
		| "opacity" => style.opacity.is_some(),
		| _ => true,
	}
}

/// Reads one selector: a tag, a class, an identifier and maybe `:hover`.
///
/// @param text - the selector, trimmed
/// @return the selector, or a sentence saying what is not supported about it
fn compound(text: &str) -> Result<Selector, String> {
	let mut selector = Selector::default();
	let mut part = String::new();
	let mut kind = ' ';

	// a small state machine rather than a split: the marks are separators and
	// part of what follows them, and `div.panel:hover` has three of them.
	for character in text.chars().chain(std::iter::once('\0')) {
		if matches!(character, '.' | '#' | ':' | '\0') {
			place(&mut selector, kind, &part, text)?;

			part.clear();
			kind = character;

			continue;
		}

		if character.is_whitespace() || character == '>' || character == '*' {
			return Err(format!(
				"`{text}` is more than one box: colby matches a tag, a class and an identifier, \
				 and has no combinators"
			));
		}

		part.push(character);
	}

	Ok(selector)
}

/// Puts one piece of a selector where it belongs.
fn place(selector: &mut Selector, kind: char, part: &str, whole: &str) -> Result<(), String> {
	if part.is_empty() {
		return Ok(());
	}

	match kind {
		| ' ' => selector.tag = part.to_ascii_lowercase(),
		| '.' =>
			if selector.class.is_empty() {
				part.clone_into(&mut selector.class);
			} else {
				return Err(format!("`{whole}` has two classes in it, and colby matches one"));
			},
		| '#' => part.clone_into(&mut selector.id),
		| ':' =>
			if part.eq_ignore_ascii_case("hover") {
				selector.hover = true;
			} else {
				return Err(format!("`{whole}` uses `:{part}`, and colby only knows `:hover`"));
			},
		| _ => return Err(format!("`{whole}` is not a selector colby reads")),
	}

	Ok(())
}

/// Reads a length: `12px`, `50%`, `auto` or a bare zero.
#[must_use]
pub fn length(text: &str) -> Option<Length> {
	let text = text.trim();

	if text.eq_ignore_ascii_case("auto") {
		return Some(Length::Auto);
	}

	if let Some(number) = text.strip_suffix('%') {
		return finite(number.trim().parse().ok()?).map(Length::Percent);
	}

	if let Some(number) = text.strip_suffix("px") {
		return finite(number.trim().parse().ok()?).map(Length::Px);
	}

	// a bare number is pixels, which CSS only allows for zero. Allowing it
	// everywhere costs nothing and is what everybody types by accident.
	finite(text.parse().ok()?).map(Length::Px)
}

/// Reads a plain number.
#[must_use]
pub fn number(text: &str) -> Option<f32> { finite(text.trim().parse().ok()?) }

/// Reads a color: `#rgb`, `#rrggbb`, `#rrggbbaa`, `rgb(..)`, `rgba(..)` or one
/// of a short list of names.
///
/// The names are the ones somebody types while roughing an interface out. A
/// long list of them is a table nobody reads; a stylesheet that has settled
/// uses hex.
#[must_use]
pub fn color(text: &str) -> Option<Color> {
	let text = text.trim();

	match text.to_ascii_lowercase().as_str() {
		| "transparent" => return Some(Color::NONE),
		| "black" => return Some(Color::from_srgb(0, 0, 0, 255)),
		| "white" => return Some(Color::from_srgb(255, 255, 255, 255)),
		| "red" => return Some(Color::from_srgb(255, 0, 0, 255)),
		| "green" => return Some(Color::from_srgb(0, 128, 0, 255)),
		| "blue" => return Some(Color::from_srgb(0, 0, 255, 255)),
		| "grey" | "gray" => return Some(Color::from_srgb(128, 128, 128, 255)),
		| _ => {},
	}

	if let Some(hex) = text.strip_prefix('#') {
		return from_hex(hex);
	}

	from_function(text)
}

/// `#rgb`, `#rrggbb` or `#rrggbbaa`.
fn from_hex(hex: &str) -> Option<Color> {
	let digits: Vec<u8> = hex
		.chars()
		.map(|character| character.to_digit(16))
		.map(|digit| u8::try_from(digit?).ok())
		.collect::<Option<Vec<u8>>>()?;

	match digits.as_slice() {
		| [red, green, blue] => Some(Color::from_srgb(red * 17, green * 17, blue * 17, 255)),
		| [red_high, red_low, green_high, green_low, blue_high, blue_low] =>
			Some(Color::from_srgb(
				red_high * 16 + red_low,
				green_high * 16 + green_low,
				blue_high * 16 + blue_low,
				255,
			)),
		| [
			red_high,
			red_low,
			green_high,
			green_low,
			blue_high,
			blue_low,
			alpha_high,
			alpha_low,
		] => Some(Color::from_srgb(
			red_high * 16 + red_low,
			green_high * 16 + green_low,
			blue_high * 16 + blue_low,
			alpha_high * 16 + alpha_low,
		)),
		| _ => None,
	}
}

/// `rgb(r, g, b)` or `rgba(r, g, b, a)`, with the alpha as a fraction.
fn from_function(text: &str) -> Option<Color> {
	let lowered = text.to_ascii_lowercase();
	let inside = lowered
		.strip_prefix("rgba(")
		.or_else(|| lowered.strip_prefix("rgb("))?
		.strip_suffix(')')?;

	let parts: Vec<&str> = inside.split(',').map(str::trim).collect();
	let channel = |index: usize| -> Option<u8> {
		let value: f32 = parts.get(index)?.parse().ok()?;

		Some(byte(value.clamp(0.0, 255.0)))
	};

	let alpha = match parts.len() {
		| 3 => 255,
		| 4 => byte(parts.get(3)?.parse::<f32>().ok()?.clamp(0.0, 1.0) * 255.0),
		| _ => return None,
	};

	Some(Color::from_srgb(channel(0)?, channel(1)?, channel(2)?, alpha))
}

/// Drops the quotation marks around a value that has them.
fn unquote(text: &str) -> &str {
	let text = text.trim();

	text.strip_prefix('"')
		.and_then(|rest| rest.strip_suffix('"'))
		.or_else(|| {
			text.strip_prefix('\'')
				.and_then(|rest| rest.strip_suffix('\''))
		})
		.unwrap_or(text)
}

/// Removes `/* ... */`, so nothing after it has to think about comments.
///
/// An unterminated comment swallows the rest of the sheet, which is what a
/// browser does and is at least easy to see.
fn strip_comments(text: &str) -> String {
	let mut out = String::with_capacity(text.len());
	let mut rest = text;

	while let Some(open) = rest.find("/*") {
		out.push_str(rest.get(..open).unwrap_or_default());

		let after = rest.get(open + 2..).unwrap_or_default();
		match after.find("*/") {
			| Some(close) => rest = after.get(close + 2..).unwrap_or_default(),
			| None => return out,
		}
	}

	out.push_str(rest);

	out
}

/// A number, unless it is not one.
fn finite(value: f32) -> Option<f32> { value.is_finite().then_some(value) }

/// A channel value from a number already clamped into range.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "every caller clamps into 0..=255 first, so there is nothing to truncate"
)]
fn byte(value: f32) -> u8 { value.round().clamp(0.0, 255.0) as u8 }

#[cfg(test)]
mod tests {
	use super::*;

	/// The style one declaration produces, and nothing about warnings.
	fn one(text: &str) -> Style { declarations(text, &mut Vec::new()) }

	#[test]
	fn a_rule_is_a_selector_and_some_declarations() {
		let sheet = parse(".panel { width: 200px; color: #fff; }");

		assert_eq!(sheet.rules.len(), 1, "one rule");
		assert_eq!(sheet.rules[0].selector.class, "panel", "claimed by class");
		assert_eq!(sheet.rules[0].style.width, Some(Length::Px(200.0)), "and it sets a width");
		assert!(sheet.warnings.is_empty(), "with nothing to complain about");
	}

	#[test]
	fn a_selector_list_becomes_one_rule_each() {
		let sheet = parse("div, .panel, #hud { opacity: 0.5; }");

		assert_eq!(sheet.rules.len(), 3, "three selectors, three rules");
		assert!(
			sheet
				.rules
				.iter()
				.all(|rule| rule.style.opacity == Some(0.5)),
			"all saying the same thing"
		);
	}

	#[test]
	fn a_hover_selector_is_read_as_one() {
		let sheet = parse("button.wide:hover { color: red; }");
		let selector = &sheet.rules[0].selector;

		assert_eq!(selector.tag, "button", "the tag");
		assert_eq!(selector.class, "wide", "the class");
		assert!(selector.hover, "and the state");
		assert_eq!(selector.specificity(), 21, "an element, a class and a pseudo-class");
	}

	#[test]
	fn a_selector_colby_cannot_match_is_a_warning_rather_than_a_wrong_match() {
		let sheet = parse(".panel .label { color: red; } .a.b { color: red; } p::first-line {}");

		assert!(sheet.rules.is_empty(), "none of the three is matched");
		assert_eq!(sheet.warnings.len(), 3, "and each says why: {:?}", sheet.warnings);
		assert!(
			sheet.warnings[0].contains("combinator"),
			"the descendant selector names what is missing: {}",
			sheet.warnings[0]
		);
	}

	#[test]
	fn a_property_nobody_implemented_is_a_warning_with_the_line_in_it() {
		let mut warnings = Vec::new();
		declarations("box-shadow: 0 0 4px black", &mut warnings);

		assert_eq!(warnings.len(), 1, "one thing was not understood");
		assert!(
			warnings[0].contains("box-shadow"),
			"and the message quotes it back: {}",
			warnings[0]
		);
	}

	#[test]
	fn a_known_property_with_an_unreadable_value_is_also_a_warning() {
		let mut warnings = Vec::new();
		let style = declarations("width: 12rem", &mut warnings);

		assert_eq!(style.width, None, "nothing was written");
		assert_eq!(
			warnings.len(),
			1,
			"and a box that is silently the wrong size is the bug this catches"
		);
	}

	#[test]
	fn lengths_come_in_three_kinds() {
		assert_eq!(length("12px"), Some(Length::Px(12.0)), "pixels");
		assert_eq!(length("50%"), Some(Length::Percent(50.0)), "percent");
		assert_eq!(length("auto"), Some(Length::Auto), "and auto");
		assert_eq!(length("0"), Some(Length::Px(0.0)), "a bare zero is pixels");
		assert_eq!(length("8"), Some(Length::Px(8.0)), "and so is a bare anything");
		assert_eq!(length("wide"), None, "a word is not a length");
	}

	#[test]
	fn hex_colors_are_read_in_all_three_lengths() {
		let short = color("#f00").expect("three digits");
		let long = color("#ff0000").expect("six digits");
		let alpha = color("#ff000080").expect("eight digits");

		assert_eq!(short, long, "`#f00` is `#ff0000`, digit for digit");
		assert!(alpha.0.w > 0.4 && alpha.0.w < 0.6, "and the last pair is opacity");
		assert_eq!(color("#ff00"), None, "four digits is not a form colby reads");
	}

	#[test]
	fn the_function_forms_are_read_too() {
		let opaque = color("rgb(255, 0, 0)").expect("three channels");
		let half = color("rgba(255, 0, 0, 0.5)").expect("and an alpha");

		assert_eq!(opaque, color("#ff0000").expect("the same color"), "the same red");
		assert!(half.0.w > 0.49 && half.0.w < 0.51, "at half opacity");
		assert_eq!(color("rgb(255, 0)"), None, "and two channels is not a color");
	}

	#[test]
	fn the_edge_shorthands_go_round_the_box_clockwise() {
		assert_eq!(one("padding: 4px").padding, Edges::all(Length::Px(4.0)), "one for all four");

		let two = one("margin: 4px 8px").margin;
		assert_eq!(two.top, Some(Length::Px(4.0)), "vertical first");
		assert_eq!(two.left, Some(Length::Px(8.0)), "then horizontal");

		let four = one("padding: 1px 2px 3px 4px").padding;
		assert_eq!(
			[four.top, four.right, four.bottom, four.left],
			[
				Some(Length::Px(1.0)),
				Some(Length::Px(2.0)),
				Some(Length::Px(3.0)),
				Some(Length::Px(4.0))
			],
			"top, right, bottom, left"
		);
	}

	#[test]
	fn one_side_can_be_named_on_its_own() {
		let style = one("margin-left: 12px");

		assert_eq!(style.margin.left, Some(Length::Px(12.0)), "the side that was named");
		assert_eq!(style.margin.right, None, "and only that one");
	}

	#[test]
	fn comments_are_removed_before_anything_else_looks_at_the_text() {
		let sheet = parse("/* the panel */ .panel { /* wide */ width: 10px; }");

		assert_eq!(sheet.rules.len(), 1, "the comments left no rules of their own");
		assert_eq!(sheet.rules[0].style.width, Some(Length::Px(10.0)), "and no properties");
		assert!(sheet.warnings.is_empty(), "nor anything unreadable");
	}

	#[test]
	fn a_font_family_keeps_its_name_and_loses_its_quotation_marks() {
		assert_eq!(
			one("font-family: \"fonts/hack\"")
				.font_family
				.as_deref(),
			Some("fonts/hack"),
			"it is an asset name, and the quotation marks are CSS punctuation"
		);
		assert_eq!(
			one("font-family: fonts/hack")
				.font_family
				.as_deref(),
			Some("fonts/hack"),
			"with or without them"
		);
	}

	#[test]
	fn a_rule_that_is_never_closed_is_reported_rather_than_swallowed() {
		let sheet = parse(".panel { width: 10px;");

		assert!(sheet.rules.is_empty(), "nothing usable came of it");
		assert_eq!(sheet.warnings.len(), 1, "and it says so: {:?}", sheet.warnings);
	}

	#[test]
	fn an_empty_sheet_is_not_a_problem() {
		let sheet = parse("");

		assert!(sheet.rules.is_empty(), "no rules");
		assert!(sheet.warnings.is_empty(), "and nothing wrong with that");
	}
}

use num_bigint::BigUint;
use num_rational::Ratio;

use std::str::FromStr;

/// A possibly big unsigned rational number
pub type BigUrational = Ratio<BigUint>;

/// A list of all possible identifiers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentKind {
    // constants
    Pi,
    Tau,
    E,
    // functions
    Sin,
    Cos,
    Tan,
    Sqrt,
}

impl FromStr for IdentKind {
    type Err = ();

    fn from_str(s: &str) -> Result<IdentKind, ()> {
        Ok(match &*s.to_ascii_lowercase() {
            "pi" => IdentKind::Pi,
            "tau" => IdentKind::Tau,
            "e" => IdentKind::E,
            "sin" | "sine" => IdentKind::Sin,
            "cos" | "cosine" => IdentKind::Cos,
            "tan" | "tangent" => IdentKind::Tan,
            "sqrt" => IdentKind::Sqrt,
            _ => return Err(()),
        })
    }
}

/// Tokens are simple things like numbers, operators, parentheses, and so on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Num { val: BigUrational, input_base: u32 },
    Ident(IdentKind),
    Plus,
    Minus,
    Times,
    Slash,
    Hat,
    OpenParen,
    CloseParen,
}

impl TokenKind {
    pub fn from_single_char(c: char) -> Option<TokenKind> {
        Some(match c {
            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,
            '*' => TokenKind::Times,
            '/' => TokenKind::Slash,
            '^' => TokenKind::Hat,
            '(' => TokenKind::OpenParen,
            ')' => TokenKind::CloseParen,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,

    /// The index of the first character of the token
    pub index: usize,
}

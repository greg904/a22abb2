mod token;

use std::iter::FusedIterator;
use std::str::FromStr;

use num_bigint::BigUint;
use num_traits::{One, Zero};

pub use self::token::*;

/// The kind of a lexer error
#[derive(Debug, PartialEq, Eq)]
pub enum LexerErrorKind {
    UnknownToken,
}

/// When the expression is malformed, the lexer will return this error.
#[derive(Debug, PartialEq, Eq)]
pub struct LexerError {
    // The error kind
    pub kind: LexerErrorKind,

    /// The index of the first character which caused the error
    pub index: usize,
}

/// A lexer reads a mathematical expression and returns a list of tokens in the
/// expression.
/// This allows us to read the expression in a simpler way later when we want
/// to parse it.
pub struct Lexer<'a> {
    expr: &'a [u8],
    index: usize,
    has_failed: bool,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer from an expression.
    pub fn new(expr: &str) -> Lexer {
        Lexer {
            expr: expr.as_bytes(),
            index: 0,
            has_failed: false,
        }
    }

    fn consume_whitespace(&mut self) {
        while self.index < self.expr.len() {
            match self.expr[self.index] as char {
                ' ' | '\n' | '\r' | '\t' => {}
                _ => break,
            }

            self.index += 1;
        }
    }

    fn try_consume_single_char_token(&mut self) -> Option<Token> {
        if self.index < self.expr.len() {
            let original_index = self.index;
            let c = self.expr[self.index] as char;

            if let Some(kind) = TokenKind::from_single_char(c) {
                // consume the character
                self.index += 1;

                return Some(Token {
                    kind,
                    index: original_index,
                });
            }
        }

        None
    }

    fn try_consume_ident(&mut self) -> Option<Result<Token, LexerError>> {
        let original_index = self.index;
        let mut ident = String::new();

        while self.index < self.expr.len() {
            let c = self.expr[self.index] as char;

            // every letter in an identifier is alphabetic
            if !c.is_ascii_alphabetic() {
                break;
            }

            ident.push(c);

            self.index += 1;
        }

        if ident.is_empty() {
            return None;
        }

        let token_kind = IdentKind::from_str(&ident)
            .map(|k| TokenKind::Ident(k))
            .unwrap_or_else(|_| TokenKind::UnknownIdent(ident));
        Some(Ok(Token {
            kind: token_kind,
            index: original_index,
        }))
    }

    fn get_base_from_char(c: char) -> Option<u32> {
        Some(match c {
            'b' => 2,
            'o' => 8,
            'd' => 10,
            'h' | 'x' => 16,
            _ => return None,
        })
    }

    fn try_consume_num(&mut self) -> Option<Token> {
        let original_index = self.index;
        let mut numer: BigUint = Zero::zero();
        let mut denom: BigUint = One::one();
        let mut has_dot = false;
        let mut has_digit = false;
        let mut base: u32 = 10;

        while self.index < self.expr.len() {
            let c = self.expr[self.index] as char;

            if let Some(digit) = c.to_digit(base) {
                numer *= base;
                numer += digit;

                if has_dot {
                    // Each time we add a number to the numerator, we need to
                    // multiply the denominator by 10 to keep the number
                    // correct.
                    denom *= base;
                }

                has_digit = true;
            } else {
                match (c, Lexer::get_base_from_char(c)) {
                    ('.', _) => {
                        // ignore dots after the first one
                        if has_dot {
                            break;
                        }

                        has_dot = true;
                    }

                    // ignore apostrophes in numbers
                    ('\'', _) => {}

                    // If no number was specified yet, the user can specify an
                    // input base. This allows them to write numbers like
                    // 0xCAFE or 00b110.
                    (_, Some(new_base)) if has_digit && numer.is_zero() => {
                        base = new_base;
                    }

                    _ => break,
                }
            }

            self.index += 1;

            // ignore whitespace between digits
            self.consume_whitespace();
        }

        if !has_digit {
            self.index = original_index;
            return None;
        }

        Some(Token {
            kind: TokenKind::Num {
                val: BigUrational::new(numer, denom),
                input_base: base,
            },
            index: original_index,
        })
    }
}

// This means that when it returns a none option, then it will keep returning
// none options.
impl<'a> FusedIterator for Lexer<'a> {}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token, LexerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_failed {
            return None;
        }

        self.consume_whitespace();

        // is there anything left?
        if self.index >= self.expr.len() {
            return None;
        }

        let original_index = self.index;
        let maybe_token = self
            .try_consume_single_char_token()
            .or_else(|| self.try_consume_num());

        // if we couldn't get a token yet, try to parse an identifier
        let maybe_token = match maybe_token {
            Some(val) => Some(val),
            None => match self.try_consume_ident() {
                Some(Ok(val)) => Some(val),
                Some(Err(err)) => {
                    self.has_failed = true;

                    // do not try another token if this didn't work
                    return Some(Err(err));
                }
                None => None,
            },
        };

        Some(maybe_token.ok_or_else(|| {
            self.has_failed = true;

            // if we didn't get any token, then it is unknown
            LexerError {
                kind: LexerErrorKind::UnknownToken,
                index: original_index,
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_handles_empty_string() {
        let mut lexer = Lexer::new("");
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn it_ignores_whitespace() {
        let mut lexer = Lexer::new("\t+ \r\n");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Plus,
                index: 1
            }))
        );
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn it_handles_single_char_tokens() {
        const EXPECTED: [TokenKind; 7] = [
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Times,
            TokenKind::Slash,
            TokenKind::Hat,
            TokenKind::OpenParen,
            TokenKind::CloseParen,
        ];

        let expected_tokens: Vec<Token> = EXPECTED
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, kind)| Token { kind, index: i })
            .collect();

        let actual_tokens: Vec<Token> = Lexer::new("+-*/^()").map(|r| r.unwrap()).collect();

        assert_eq!(actual_tokens, expected_tokens);
    }

    fn one_two_three() -> TokenKind {
        TokenKind::Num {
            val: BigUrational::from(BigUint::from(123u32)),
            input_base: 10,
        }
    }

    fn dot_one_two_three() -> TokenKind {
        TokenKind::Num {
            val: BigUrational::new(BigUint::from(123u32), BigUint::from(1000u32)),
            input_base: 10,
        }
    }

    #[test]
    fn it_handles_integer_numbers() {
        let mut lexer = Lexer::new("123");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: one_two_three(),
                index: 0
            }))
        );
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new("1 2'3");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: one_two_three(),
                index: 0
            }))
        );
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new("-123");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Minus,
                index: 0
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: one_two_three(),
                index: 1
            }))
        );
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new("+123");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Plus,
                index: 0
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: one_two_three(),
                index: 1
            }))
        );
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn it_handles_numbers_with_decimal_points() {
        let mut lexer = Lexer::new("123.");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: one_two_three(),
                index: 0
            }))
        );
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new(".123");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: dot_one_two_three(),
                index: 0
            }))
        );
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new(".1 2'3");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: dot_one_two_three(),
                index: 0
            }))
        );
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new("-.123");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Minus,
                index: 0
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: dot_one_two_three(),
                index: 1
            }))
        );
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new("+.123");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Plus,
                index: 0
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: dot_one_two_three(),
                index: 1
            }))
        );
        assert_eq!(lexer.next(), None);

        let mut lexer = Lexer::new("+0.");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Plus,
                index: 0
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Num {
                    val: Zero::zero(),
                    input_base: 10
                },
                index: 1
            }))
        );
        assert_eq!(lexer.next(), None);

        // must fail
        let mut lexer = Lexer::new("+.");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Plus,
                index: 0
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Err(LexerError {
                kind: LexerErrorKind::UnknownToken,
                index: 1
            }))
        );
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn it_handles_other_number_bases() {
        // Here, we test if the lexer:
        // - stops the current number when a digit is not in a base (eg. 0o9)
        // - handles whitespace and apostrophes correctly between base specifiers
        // - handles non-integer numbers in other bases
        // - ignores all of the zeros before the base specifier
        let mut lexer = Lexer::new("0b1012+000 o'47.48+0'h 9F+00'x'9F. E");

        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Num {
                    val: BigUrational::from(BigUint::from(0b101u32)),
                    input_base: 2
                },
                index: 0
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Num {
                    val: BigUrational::from(BigUint::from(2u32)),
                    input_base: 10
                },
                index: 5
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Plus,
                index: 6
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Num {
                    val: BigUrational::new(BigUint::from(0o474u32), BigUint::from(8u32)),
                    input_base: 8
                },
                index: 7
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Num {
                    val: BigUrational::from(BigUint::from(8u32)),
                    input_base: 10
                },
                index: 17
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Plus,
                index: 18
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Num {
                    val: BigUrational::from(BigUint::from(0x9Fu32)),
                    input_base: 16
                },
                index: 19
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Plus,
                index: 25
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Num {
                    val: BigUrational::new(BigUint::from(0x9FEu32), BigUint::from(16u32)),
                    input_base: 16
                },
                index: 26
            }))
        );
    }

    #[test]
    fn it_handles_constants() {
        let mut lexer = Lexer::new("pi*tau+zzz");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Ident(IdentKind::Pi),
                index: 0
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Times,
                index: 2
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Ident(IdentKind::Tau),
                index: 3
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Plus,
                index: 6
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::UnknownIdent("zzz".to_string()),
                index: 7
            }))
        );
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn it_handles_functions() {
        let mut lexer = Lexer::new("sin(cos sqrt(1))");
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Ident(IdentKind::Sin),
                index: 0
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::OpenParen,
                index: 3
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Ident(IdentKind::Cos),
                index: 4
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Ident(IdentKind::Sqrt),
                index: 8
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::OpenParen,
                index: 12
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::Num {
                    val: One::one(),
                    input_base: 10
                },
                index: 13
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::CloseParen,
                index: 14
            }))
        );
        assert_eq!(
            lexer.next(),
            Some(Ok(Token {
                kind: TokenKind::CloseParen,
                index: 15
            }))
        );
        assert_eq!(lexer.next(), None);
    }
}

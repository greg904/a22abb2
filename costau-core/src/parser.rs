use super::lexer::{IdentKind, Token, TokenKind};
use super::node::{ConstKind, Node};

use num_rational::BigRational;

#[derive(PartialEq, Eq)]
enum StopPolicy {
    IfWeaker(Power),
    IfWeakerOrEqual(Power),
    Never,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Power {
    CloseParen,
    Add,
    Mul,
    Exp,
    ImplicitMul,
}

/// A parser converts a list of tokens into an AST (abstract syntax tree).
pub struct Parser<'a> {
    tokens: &'a [Token],
    index: usize,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ParseError {
    EarlyEof,
    UnexpectedToken { index: usize },
    UnmatchedParen,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &[Token]) -> Parser {
        Parser { tokens, index: 0 }
    }

    fn parse_nud(&mut self) -> Result<Node, ParseError> {
        if self.index >= self.tokens.len() {
            return Err(ParseError::EarlyEof);
        }

        let original_index = self.index;
        let token = self.tokens[self.index].clone();
        self.index += 1;

        Ok(match token.kind {
            TokenKind::Num { val, input_base } => {
                let (numer, denom) = val.into();
                Node::Num {
                    val: BigRational::new(numer.into(), denom.into()),
                    input_base: Some(input_base),
                }
            }
            TokenKind::Ident(kind) => match kind {
                // constants
                IdentKind::Pi => Node::Const(ConstKind::Pi),
                IdentKind::Tau => Node::Const(ConstKind::Tau),
                IdentKind::E => Node::Const(ConstKind::E),
                // functions
                _ => {
                    let param = Box::new(self.parse_nud()?);
                    match kind {
                        IdentKind::Sin => Node::Sin(param),
                        IdentKind::Cos => Node::Cos(param),
                        IdentKind::Tan => Node::Tan(param),
                        IdentKind::Sqrt => param.sqrt(),
                        IdentKind::Cbrt => param.cbrt(),
                        _ => unreachable!(),
                    }
                }
            },
            TokenKind::UnknownIdent(s) => Node::UnknownConst(s),

            TokenKind::Minus => -self.parse_nud()?,
            TokenKind::Plus => self.parse_nud()?,
            TokenKind::OpenParen => {
                let expr = self.parse_range(&StopPolicy::IfWeakerOrEqual(Power::CloseParen))?;
                let is_closed = self
                    .tokens
                    .get(self.index)
                    .map_or(false, |t| t.kind == TokenKind::CloseParen);

                // closing parentheses are optional
                if is_closed {
                    // consume the parenthesis
                    self.index += 1;
                }

                expr
            }

            _ => {
                self.index = original_index;
                return Err(ParseError::UnexpectedToken { index: token.index });
            }
        })
    }

    fn parse_led(&mut self, left: Node) -> Result<Node, ParseError> {
        if self.index >= self.tokens.len() {
            return Err(ParseError::EarlyEof);
        }

        let original_index = self.index;
        let token = self.tokens[self.index].clone();
        self.index += 1;

        Ok(match token.kind {
            // left associativity
            TokenKind::Plus => left + self.parse_range(&StopPolicy::IfWeakerOrEqual(Power::Add))?,
            TokenKind::Minus => {
                left - self.parse_range(&StopPolicy::IfWeakerOrEqual(Power::Add))?
            }
            TokenKind::Slash => {
                left / self.parse_range(&StopPolicy::IfWeakerOrEqual(Power::Mul))?
            }

            // right associativity: 1^2^3 is parsed as exp(1, exp(2, 3)), not exp(exp(1, 2), 3)
            TokenKind::Hat => Node::Exp(
                Box::new(left),
                Box::new(self.parse_range(&StopPolicy::IfWeaker(Power::Exp))?),
            ),

            // implicit or explicit multiplication
            _ => {
                //
                if token.kind != TokenKind::Times {
                    // do not consume the token if it is implicit multiplication
                    self.index = original_index;
                }
                left * self.parse_range(&StopPolicy::IfWeakerOrEqual(Power::Mul))?
            }
        })
    }

    fn parse_range(&mut self, policy: &StopPolicy) -> Result<Node, ParseError> {
        let mut node = self.parse_nud()?;
        let original_index = self.index;

        'parse: while self.index < self.tokens.len() {
            if policy != &StopPolicy::Never {
                let peek = self.tokens[self.index].clone();

                // There are multiple modes to tell the parse when to stop.
                // For example, if we are parsing an expression inside of
                // parentheses, we want to stop when we encounter a closing
                // parenthesis.
                match &policy {
                    StopPolicy::IfWeaker(..) | StopPolicy::IfWeakerOrEqual(..) => {
                        let maybe_power = match peek.kind {
                            TokenKind::CloseParen => Some(Power::CloseParen),
                            TokenKind::Plus | TokenKind::Minus => Some(Power::Add),
                            TokenKind::Times | TokenKind::Slash => Some(Power::Mul),
                            TokenKind::Hat => Some(Power::Exp),

                            // implicit multiplication
                            _ => Some(Power::ImplicitMul),
                        };

                        if let Some(ref power) = maybe_power {
                            let (min_power, stop_if_equal) = match &policy {
                                StopPolicy::IfWeaker(val) => (val, false),
                                StopPolicy::IfWeakerOrEqual(val) => (val, true),
                                _ => unreachable!(),
                            };

                            if power < min_power || (stop_if_equal && power == min_power) {
                                break 'parse;
                            }
                        }
                    }
                    StopPolicy::Never => unreachable!(),
                };
            }

            node = match self.parse_led(node) {
                Ok(val) => val,
                Err(err) => {
                    self.index = original_index;
                    return Err(err);
                }
            };
        }

        Ok(node)
    }

    pub fn parse(mut self) -> Result<Node, ParseError> {
        self.parse_range(&StopPolicy::Never)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::lexer::Lexer;
    use num_traits::One;

    #[test]
    fn it_handles_precedence_correctly_with_functions() {
        let tokens: Vec<Token> = Lexer::new("sin(cos sqrt(1))").map(|x| x.unwrap()).collect();
        let parser = Parser::new(&tokens);
        let root_node = parser.parse().unwrap();
        assert_eq!(
            root_node,
            Node::Sin(Box::new(Node::Cos(Box::new(
                Node::Num {
                    val: One::one(),
                    input_base: Some(10)
                }
                .sqrt()
            ))))
        );
    }

    #[test]
    fn it_follows_the_convention_of_higher_precedence_for_implicit_multiplication() {
        // The precendence of implicit multiplication is not clearly defined but
        // I believe that when the user enters `1/2pi`, they mean `1/(2pi)`
        // instead of `pi/2`.
        let tokens: Vec<Token> = Lexer::new("1/2pi").map(|x| x.unwrap()).collect();
        let parser = Parser::new(&tokens);
        let root_node = parser.parse().unwrap();

        let one = Node::Num {
            val: One::one(),
            input_base: Some(10)
        };
        let two = Node::Num {
            val: BigRational::from_integer(2.into()),
            input_base: Some(10)
        };
        assert_eq!(
            root_node,
            one / (two * Node::Const(ConstKind::Pi))
        );
    }
}

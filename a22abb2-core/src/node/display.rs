use num_traits::{One, Signed};
use std::fmt;
use std::fmt::{Display, Write};

use super::util::is_minus_one;
use super::{ConstKind, Node};

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum NodePriority {
    AddOrSub,
    MulOrDiv,
    Exp,
    Value,
}

fn get_node_priority(node: &Node) -> NodePriority {
    match node {
        Node::Const(_) | Node::UnknownConst(_) => NodePriority::Value,
        Node::Num { val, .. } => {
            if val.denom().is_one() {
                NodePriority::Value
            } else {
                // it will be displayed as a fraction with a division sign
                NodePriority::MulOrDiv
            }
        }
        Node::Sum(_) => NodePriority::AddOrSub,
        Node::Product(_) => NodePriority::MulOrDiv,
        Node::Exp(_, b) => {
            if is_minus_one(b) {
                // will show as inverse
                NodePriority::MulOrDiv
            } else {
                NodePriority::Exp
            }
        }
        // functions
        Node::Sin(_) | Node::Cos(_) | Node::Tan(_) => NodePriority::Exp,
    }
}

fn write_with_paren(
    f: &mut fmt::Formatter<'_>,
    node: &Node,
    curr_prio: NodePriority,
    left_assoc: bool,
    needs_separation: bool,
) -> fmt::Result {
    let needs_paren = if left_assoc {
        // mul(1,mul(2,3)) => 1*2*3
        get_node_priority(node) < curr_prio
    } else {
        // pow(1,pow(2,3)) => 1^(2^3)
        get_node_priority(node) <= curr_prio
    };
    if needs_paren {
        f.write_char('(')?;
    } else if needs_separation {
        f.write_char(' ')?;
    }
    node.fmt(f)?;
    if needs_paren {
        f.write_char(')')?;
    }
    Ok(())
}

fn write_func(f: &mut fmt::Formatter<'_>, name: &str, inner: &Node) -> fmt::Result {
    f.write_str(name)?;
    write_with_paren(f, inner, NodePriority::Exp, true, true)
}

impl Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Const(kind) => match kind {
                ConstKind::Pi => write!(f, "pi"),
                ConstKind::Tau => write!(f, "tau"),
                ConstKind::E => write!(f, "e"),
            },
            Node::UnknownConst(s) => write!(f, "{}", s),
            Node::Num { val, input_base } => {
                let input_base = input_base.unwrap_or(10);
                match input_base {
                    2 if val.is_integer() => write!(f, "{:#b}", val.numer()),
                    8 if val.is_integer() => write!(f, "{:#o}", val.numer()),
                    10 => write!(f, "{}", val),
                    16 if val.is_integer() => write!(f, "{:#X}", val.numer()),
                    input_base => {
                        eprintln!("warning: cannot print BigRational in base {} yet", input_base);
                        write!(f, "{}", val)
                    }
                }
            }
            Node::Sum(children) => {
                let mut first = true;
                for child in children {
                    if first {
                        first = false;
                    } else {
                        // detect subtraction
                        if let Node::Product(c_children) = child {
                            if c_children.len() == 2 {
                                let mut is_done = false;
                                for i in 0..=1 {
                                    if is_minus_one(&c_children[i]) {
                                        // directly output "- x" instead of "+ (-1) * x"
                                        write!(f, " - ")?;
                                        write_with_paren(
                                            f,
                                            &c_children[1 - i],
                                            NodePriority::AddOrSub,
                                            false,
                                            false,
                                        )?;
                                        is_done = true;
                                        break;
                                    }
                                }
                                if is_done {
                                    continue;
                                }
                            }
                        } else if let Node::Num { val, input_base } = child {
                            if val.is_negative() {
                                // directly output "- x" instead of "+ -x"
                                write!(f, " - ")?;
                                let fake_node = Node::Num {
                                    val: -val, // remove negative sign
                                    input_base: *input_base,
                                };
                                fake_node.fmt(f)?;
                                continue;
                            }
                        }
                        write!(f, " + ")?;
                    };
                    write_with_paren(f, child, get_node_priority(self), true, false)?;
                }
                Ok(())
            }
            Node::Product(children) => {
                let mut first = true;
                for child in children {
                    if first {
                        first = false;
                    } else {
                        // detect division
                        if let Node::Exp(a, b) = child {
                            if is_minus_one(b) {
                                // directly output "/ x" instead of "* 1/x"
                                write!(f, " / ")?;
                                write_with_paren(f, a, NodePriority::MulOrDiv, false, false)?;
                                continue;
                            }
                        }
                        write!(f, " * ")?;
                    }
                    write_with_paren(f, child, get_node_priority(self), true, false)?;
                }
                Ok(())
            }
            Node::Exp(a, b) => {
                if is_minus_one(b) {
                    // a^-1 = 1/a
                    write!(f, "1/")?;
                    write_with_paren(f, a, NodePriority::MulOrDiv, true, false)
                } else {
                    write_with_paren(f, a, NodePriority::Exp, false, false)?;
                    f.write_char('^')?;
                    write_with_paren(f, b, NodePriority::Exp, false, false)
                }
            }
            // functions
            Node::Sin(inner) => write_func(f, "sin", inner),
            Node::Cos(inner) => write_func(f, "cos", inner),
            Node::Tan(inner) => write_func(f, "tan", inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::lexer::{Lexer, Token};
    use crate::parser::Parser;

    #[test]
    fn it_formats_a_node_correctly() {
        const CASES: [&str; 7] = [
            // easy
            "1+2",
            "1*3+5",
            "1^pi",
            // weird cases
            "1/(2/3)",
            "pi/2e",
            "sin cos 2",
            // number base
            "0xFF/0b10*sin(2)",
        ];
        for c in &CASES {
            let tokens: Vec<Token> = Lexer::new(c).map(|x| x.unwrap()).collect();
            let root_node = Parser::new(&tokens).parse().unwrap();

            // format it and re-parse it to check if it changed
            let formatted = root_node.to_string();
            let new_tokens: Vec<Token> = Lexer::new(&formatted).map(|x| x.unwrap()).collect();
            let new_root_node = Parser::new(&new_tokens).parse().unwrap();

            let ground_truth = root_node.eval().unwrap();
            let result_from_formatted = new_root_node.eval().unwrap();
            assert!((result_from_formatted.val - ground_truth.val).abs() < 0.001);
            assert_eq!(
                result_from_formatted.display_base,
                ground_truth.display_base
            );
        }
    }
}

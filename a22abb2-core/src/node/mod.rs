mod simplify;
pub use self::simplify::*;

use crate::ratio2flt::ratio_to_f64;
use crate::EvalResult;

use std::f64::consts::{E, PI};
use std::fmt;
use std::fmt::{Display, Write};
use std::ops::{Add, Mul};

use num_rational::BigRational;
use num_traits::{One, Zero};

/// A constant in mathematics
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ConstKind {
    Pi,
    Tau,
    E,
}

/// A kind of operator that can take multiple children
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum VarOpKind {
    Add,
    Mul,
}

impl VarOpKind {
    pub fn identity_f64(self) -> f64 {
        match self {
            VarOpKind::Add => 0.0,
            VarOpKind::Mul => 1.0,
        }
    }

    pub fn identity_bigr(self) -> BigRational {
        match self {
            VarOpKind::Add => Zero::zero(),
            VarOpKind::Mul => One::one(),
        }
    }

    pub fn eval_f64_fn(self) -> &'static dyn Fn(f64, f64) -> f64 {
        match self {
            VarOpKind::Add => &Add::add,
            VarOpKind::Mul => &Mul::mul,
        }
    }

    pub fn eval_bigr_fn(self) -> &'static dyn Fn(BigRational, BigRational) -> BigRational {
        match self {
            VarOpKind::Add => &Add::add,
            VarOpKind::Mul => &Mul::mul,
        }
    }

    fn compress(self, node: Node, count: Node) -> Node {
        match self {
            VarOpKind::Add => Node::mul(node, count),
            VarOpKind::Mul => Node::Exp(Box::new(node), Box::new(count)),
        }
    }
}

/// A node is an operation in the AST (abstract syntax tree).
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Node {
    Const(ConstKind),
    Num {
        /// The number the node represents
        val: BigRational,

        /// The base the number was written in by the user, if it was written
        /// by the user
        input_base: Option<u32>,
    },
    Inverse(Box<Node>),
    VarOp {
        kind: VarOpKind,
        children: Vec<Node>,
    },
    Exp(Box<Node>, Box<Node>),
    // functions
    Sin(Box<Node>),
    Cos(Box<Node>),
    Tan(Box<Node>),
}

impl Node {
    /// Approximates the node value.
    pub fn eval(&self) -> EvalResult {
        match self {
            Node::Const(kind) => EvalResult {
                val: match kind {
                    ConstKind::Pi => PI,
                    ConstKind::Tau => PI * 2.0,
                    ConstKind::E => E,
                },
                display_base: None,
            },
            Node::Num { val, input_base } => EvalResult {
                val: ratio_to_f64(&val),
                display_base: input_base.clone(),
            },
            Node::Inverse(inner) => inner.eval_map(|x| 1.0 / x, true),
            Node::VarOp { kind, children } => Node::eval_var_op(children.into_iter(), kind.clone()),
            Node::Exp(a, b) => {
                let a = a.eval();
                let b = b.eval();
                EvalResult {
                    val: a.val.powf(b.val),
                    display_base: Node::get_op_result_base(a.display_base, b.display_base),
                }
            },
            Node::Sin(inner) => inner.eval_map(|x| x.sin(), false),
            Node::Cos(inner) => inner.eval_map(|x| x.cos(), false),
            Node::Tan(inner) => inner.eval_map(|x| x.tan(), false),
        }
    }

    fn eval_var_op<'a, I>(children: I, kind: VarOpKind) -> EvalResult
    where
        I: Iterator<Item = &'a Node>,
    {
        let mut result = kind.identity_f64();
        let mut result_base = None;

        for child in children {
            let child = child.eval();
            result = kind.eval_f64_fn()(result, child.val);
            result_base = Node::get_op_result_base(result_base, child.display_base);
        }

        EvalResult {
            val: result,
            display_base: result_base,
        }
    }

    fn eval_map<F: Fn(f64) -> f64>(&self, f: F, keep_base: bool) -> EvalResult {
        let original = self.eval();
        EvalResult {
            val: f(original.val),
            display_base: if keep_base { original.display_base } else { None },
        }
    }

    fn get_op_result_base(a_base: Option<u32>, b_base: Option<u32>) -> Option<u32> {
        match (a_base, b_base) {
            (Some(val), None) | (None, Some(val)) => Some(val),

            // prefer the more interesting bases
            (Some(10), Some(other)) | (Some(other), Some(10)) => Some(other),
            (Some(2), _) | (_, Some(2)) => Some(2),

            (Some(a), Some(_)) => Some(a), // prefer the base of the first term
            (None, None) => None,
        }
    }

    fn zero() -> Node {
        Node::Num {
            val: Zero::zero(),
            input_base: None,
        }
    }

    fn one() -> Node {
        Node::Num {
            val: One::one(),
            input_base: None,
        }
    }

    fn two() -> Node {
        Node::Num {
            val: BigRational::from_integer(2.into()),
            input_base: None,
        }
    }

    fn three() -> Node {
        Node::Num {
            val: BigRational::from_integer(3.into()),
            input_base: None,
        }
    }

    fn minus_one() -> Node {
        Node::Num {
            val: -BigRational::one(),
            input_base: None,
        }
    }

    pub fn add(a: Node, b: Node) -> Node {
        Node::op(VarOpKind::Add, a, b)
    }

    pub fn sub(a: Node, b: Node) -> Node {
        Node::add(a, Node::opposite(b))
    }

    pub fn mul(a: Node, b: Node) -> Node {
        Node::op(VarOpKind::Mul, a, b)
    }

    pub fn div(a: Node, b: Node) -> Node {
        Node::mul(a, Node::Inverse(Box::new(b)))
    }

    fn op(kind: VarOpKind, a: Node, b: Node) -> Node {
        Node::VarOp {
            kind,
            children: vec![a, b],
        }
    }

    pub fn opposite(self) -> Node {
        Node::mul(Node::minus_one(), self)
    }

    pub fn inverse(self) -> Node {
        Node::Inverse(Box::new(self))
    }

    pub fn sqrt(self) -> Node {
        Node::Exp(
            Box::new(self),
            Box::new(Node::two().inverse()),
        )
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum NodePriority {
    AddOrSub,
    MulOrDiv,
    ExpOrValue,
}

fn get_node_priority(node: &Node) -> NodePriority {
    match node {
        Node::Const(_) => NodePriority::ExpOrValue,
        Node::Num { val, .. } => {
            if val.denom().is_one() {
                NodePriority::ExpOrValue
            } else {
                // it will be displayed as a fraction with a division sign
                NodePriority::MulOrDiv
            }
        },
        Node::Inverse(_) => NodePriority::MulOrDiv,
        Node::VarOp { kind, .. } => match kind {
            VarOpKind::Add => NodePriority::AddOrSub,
            VarOpKind::Mul => NodePriority::MulOrDiv,
        },
        Node::Exp(_, _) => NodePriority::ExpOrValue,
        // functions
        Node::Sin(_) | Node::Cos(_) | Node::Tan(_) => NodePriority::ExpOrValue,
    }
}

fn write_with_paren(f: &mut fmt::Formatter<'_>, node: &Node, curr_prio: NodePriority, left_assoc: bool, needs_separation: bool) -> fmt::Result {
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
    write_with_paren(f, inner, NodePriority::ExpOrValue, true, true)
}

fn is_baseless_minus_one(node: &Node) -> bool {
    if let Node::Num { val, input_base } = node {
        return ((val.denom().is_one() && *val.numer() == (-1).into()) ||
            (*val.denom() == (-1).into() && val.numer().is_one())) &&
            input_base.is_none();
    }
    return false;
}

impl Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Const(kind) => match kind {
                ConstKind::Pi => write!(f, "pi"),
                ConstKind::Tau => write!(f, "tau"),
                ConstKind::E => write!(f, "e"),
            },
            Node::Num { val, input_base } => {
                let input_base = input_base.unwrap_or(10);
                match input_base {
                    2 if val.is_integer() => write!(f, "{:#b}", val.numer()),
                    8 if val.is_integer() => write!(f, "{:#o}", val.numer()),
                    10 => write!(f, "{}", val),
                    16 if val.is_integer() => write!(f, "{:#X}", val.numer()),
                    _ => todo!("print in bases other than 10 for decimal number"),
                }
            },
            Node::Inverse(inner) => {
                write!(f, "1/")?;
                write_with_paren(f, inner, get_node_priority(self), true, false)
            },
            Node::VarOp { kind, children } => {
                let mut first = true;
                for child in children {
                    if first {
                        first = false;
                    } else {
                        let op_char = match kind {
                            VarOpKind::Add => {
                                // detect subtraction
                                if let Node::VarOp { kind: VarOpKind::Mul, children: c_children } = child {
                                    if c_children.len() == 2 {
                                        let mut is_done = false;
                                        for i in 0..=1 {
                                            if is_baseless_minus_one(&c_children[i]) {
                                                // directly output "- x" instead of "+ (-1) * x"
                                                write!(f, " - ")?;
                                                write_with_paren(f, &c_children[1 - i], NodePriority::AddOrSub, false, false)?;
                                                is_done = true;
                                                break;
                                            }
                                        }
                                        if is_done {
                                            continue;
                                        }
                                    }
                                }
                                '+'
                            },
                            VarOpKind::Mul => {
                                // detect division
                                if let Node::Inverse(x) = child {
                                    // directly output "/ x" instead of "* 1/x"
                                    write!(f, " / ")?;
                                    write_with_paren(f, x, NodePriority::MulOrDiv, false, false)?;
                                    continue;
                                } else {
                                    '*'
                                }
                            }
                        };
                        write!(f, " {} ", op_char)?;
                    }
                    write_with_paren(f, child, get_node_priority(self), true, false)?;
                }
                Ok(())
            },
            Node::Exp(a, b) => {
                write_with_paren(f, a, NodePriority::ExpOrValue, false, false)?;
                f.write_char('^')?;
                write_with_paren(f, b, NodePriority::ExpOrValue, false, false)?;
                Ok(())
            },
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
            let tokens: Vec<Token> = Lexer::new(c)
                .map(|x| x.unwrap()).collect();
            let root_node = Parser::new(&tokens).parse().unwrap();

            // format it and re-parse it to check if it changed
            let formatted = root_node.to_string();
            let new_tokens: Vec<Token> = Lexer::new(&formatted)
                .map(|x| x.unwrap()).collect();
            let new_root_node = Parser::new(&new_tokens).parse().unwrap();

            let ground_truth = root_node.eval();
            let result_from_formatted = new_root_node.eval();
            assert!((result_from_formatted.val - ground_truth.val).abs() < 0.001);
            assert_eq!(result_from_formatted.display_base, ground_truth.display_base);
        }
    }
}

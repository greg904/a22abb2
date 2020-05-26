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
    pub fn eval(self) -> EvalResult {
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
                display_base: input_base,
            },
            Node::Inverse(inner) => inner.eval_map(|x| 1.0 / x),
            Node::VarOp { kind, children } => Node::eval_var_op(children.into_iter(), kind),
            Node::Exp(a, b) => {
                let a = a.eval();
                let b = b.eval();
                EvalResult {
                    val: a.val.powf(b.val),
                    display_base: Node::get_op_result_base(a.display_base, b.display_base),
                }
            },
            Node::Sin(inner) => inner.eval_map(|x| x.sin()),
            Node::Cos(inner) => inner.eval_map(|x| x.cos()),
            Node::Tan(inner) => inner.eval_map(|x| x.tan()),
        }
    }

    fn eval_var_op<I>(children: I, kind: VarOpKind) -> EvalResult
    where
        I: Iterator<Item = Node>,
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

    fn eval_map<F: Fn(f64) -> f64>(self, f: F) -> EvalResult {
        let original = self.eval();
        EvalResult {
            val: f(original.val),
            display_base: original.display_base,
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

    pub fn zero() -> Node {
        Node::Num {
            val: Zero::zero(),
            input_base: None,
        }
    }

    pub fn one() -> Node {
        Node::Num {
            val: One::one(),
            input_base: None,
        }
    }

    pub fn minus_one() -> Node {
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

    pub fn opposite(inner: Node) -> Node {
        Node::mul(Node::minus_one(), inner)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum NodePriority {
    Add,
    Mul,
    Exp,
    Value,
}

fn get_node_priority(node: &Node) -> NodePriority {
    match node {
        Node::Const(_) => NodePriority::Value,
        Node::Num { val, .. } => {
            if val.denom().is_one() {
                NodePriority::Value
            } else {
                // it will be displayed as a fraction with a division sign
                NodePriority::Mul
            }
        },
        Node::Inverse(_) => NodePriority::Mul,
        Node::VarOp { kind, .. } => match kind {
            VarOpKind::Add => NodePriority::Add,
            VarOpKind::Mul => NodePriority::Mul,
        },
        Node::Exp(_, _) => NodePriority::Exp,
        // functions
        Node::Sin(_) | Node::Cos(_) | Node::Tan(_) => NodePriority::Value,
    }
}

fn write_with_paren(f: &mut fmt::Formatter<'_>, node: &Node, curr_prio: NodePriority, right_assoc: bool, needs_separation: bool) -> fmt::Result {
    let needs_paren = if right_assoc {
        // pow(1,pow(2,3)) => 1^(2^3)
        get_node_priority(node) <= curr_prio
    } else {
        // mul(1,mul(2,3)) => 1*2*3
        get_node_priority(node) < curr_prio
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
    write_with_paren(f, inner, NodePriority::Value, false, true)
}

impl Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Const(kind) => match kind {
                ConstKind::Pi => write!(f, "pi"),
                ConstKind::Tau => write!(f, "tau"),
                ConstKind::E => write!(f, "e"),
            },
            // TODO: print in correct base
            Node::Num { val, input_base: _ } => write!(f, "{}", val),

            Node::Inverse(inner) => {
                write!(f, "1/")?;
                write_with_paren(f, inner, get_node_priority(self), false, false)
            },
            Node::VarOp { kind, children } => {
                let mut first = true;
                for child in children {
                    if first {
                        first = false;
                    } else {
                        let op_char = match kind {
                            VarOpKind::Add => '+',
                            VarOpKind::Mul => '*',
                        };
                        if *kind == VarOpKind::Mul {
                            if let Node::Inverse(x) = child {
                                // directly output "/ x" instead of "* 1/x"
                                write!(f, " / ")?;
                                write_with_paren(f, x, NodePriority::Mul, false, false)?;
                                continue;
                            }
                        }
                        
                        write!(f, " {} ", op_char)?;
                    }
                    write_with_paren(f, child, get_node_priority(self), false, false)?;
                }
                Ok(())
            },
            Node::Exp(a, b) => {
                write_with_paren(f, a, NodePriority::Exp, true, false)?;
                f.write_char('^')?;
                write_with_paren(f, b, NodePriority::Exp, true, false)?;
                Ok(())
            },
            // functions
            Node::Sin(inner) => write_func(f, "sin", inner),
            Node::Cos(inner) => write_func(f, "cos", inner),
            Node::Tan(inner) => write_func(f, "tan", inner),
        }
    }
}

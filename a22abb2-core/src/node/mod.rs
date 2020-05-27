mod display;
mod simplify;
pub use self::simplify::*;
mod util;

use num_rational::BigRational;
use num_traits::{One, Zero};
use std::f64::consts::{E, PI};
use std::ops::*;

use crate::ratio2flt::ratio_to_f64;
use crate::EvalResult;

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

    pub fn eval_f64_fn(self) -> &'static dyn Fn(f64, f64) -> f64 {
        match self {
            VarOpKind::Add => &Add::add,
            VarOpKind::Mul => &Mul::mul,
        }
    }

    // TODO: move the following functions to `simplify.rs`
    pub fn identity_bigr(self) -> BigRational {
        match self {
            VarOpKind::Add => Zero::zero(),
            VarOpKind::Mul => One::one(),
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
            VarOpKind::Add => node * count,
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
                display_base: *input_base,
            },
            Node::VarOp { kind, children } => Node::eval_var_op(children.iter(), *kind),
            Node::Exp(a, b) => {
                let a = a.eval();
                let b = b.eval();
                EvalResult {
                    val: a.val.powf(b.val),
                    display_base: Node::get_op_result_base(a.display_base, b.display_base),
                }
            }
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
            display_base: if keep_base {
                original.display_base
            } else {
                None
            },
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

    pub fn inverse(self) -> Node {
        Node::Exp(Box::new(self), Box::new(Node::minus_one()))
    }

    pub fn sqrt(self) -> Node {
        Node::Exp(Box::new(self), Box::new(Node::two().inverse()))
    }

    pub fn sin(self) -> Node {
        Node::Sin(Box::new(self))
    }

    pub fn cos(self) -> Node {
        Node::Cos(Box::new(self))
    }

    pub fn tan(self) -> Node {
        Node::Tan(Box::new(self))
    }
}

impl Add for Node {
    type Output = Node;

    fn add(self, rhs: Self) -> Self::Output {
        Node::VarOp {
            kind: VarOpKind::Add,
            children: vec![self, rhs],
        }
    }
}

impl Neg for Node {
    type Output = Node;

    fn neg(self) -> Self::Output {
        Node::minus_one() * self
    }
}

impl Sub for Node {
    type Output = Node;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Mul for Node {
    type Output = Node;

    fn mul(self, rhs: Self) -> Self::Output {
        Node::VarOp {
            kind: VarOpKind::Mul,
            children: vec![self, rhs],
        }
    }
}

impl Div for Node {
    type Output = Node;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

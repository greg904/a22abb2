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
use self::util::{fold_nodes, get_op_result_base};

/// A constant in mathematics
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ConstKind {
    Pi,
    Tau,
    E,
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
    Sum(Vec<Node>),
    Product(Vec<Node>),
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
            Node::Sum(children) => fold_nodes(children.iter(), 0.0, Add::add),
            Node::Product(children) => fold_nodes(children.iter(), 1.0, Mul::mul),
            Node::Exp(a, b) => {
                let a = a.eval();
                let b = b.eval();
                if b.val == 0.0 {
                    return EvalResult {
                        val: 1.0,
                        display_base: a.display_base,
                    };
                } else if b.val == 1.0 {
                    return EvalResult {
                        val: a.val,
                        display_base: a.display_base,
                    };
                } else if b.val == -1.0 {
                    return EvalResult {
                        val: 1.0 / a.val,
                        display_base: a.display_base,
                    };
                }
                EvalResult {
                    val: a.val.powf(b.val),
                    display_base: get_op_result_base(a.display_base, b.display_base),
                }
            }
            Node::Sin(inner) => inner.eval_map(f64::sin, false),
            Node::Cos(inner) => inner.eval_map(f64::cos, false),
            Node::Tan(inner) => inner.eval_map(f64::tan, false),
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
        Node::Sum(vec![self, rhs])
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
        Node::Product(vec![self, rhs])
    }
}

impl Div for Node {
    type Output = Node;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

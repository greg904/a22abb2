mod display;
mod eval;
mod simplify;
pub(crate) mod util;

use num_rational::BigRational;
use std::ops::*;

use self::eval::*;
use self::simplify::*;
use self::util::common;

/// A constant in mathematics
#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum ConstKind {
    Pi,
    Tau,
    E,
}

/// A node is an operation in the AST (abstract syntax tree).
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Node {
    Const(ConstKind),
    UnknownConst(String),
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
    pub fn eval(&self) -> Result<EvalSuccess, EvalError> {
        eval(self)
    }

    /// Simplifies the node.
    pub fn simplify(self) -> Result<SimplifySuccess, SimplifyError> {
        simplify(self)
    }

    pub fn inverse(self) -> Node {
        Node::Exp(Box::new(self), Box::new(common::minus_one()))
    }

    pub fn sqr(self) -> Node {
        Node::Exp(Box::new(self), Box::new(common::two()))
    }

    pub fn sqrt(self) -> Node {
        Node::Exp(Box::new(self), Box::new(common::two().inverse()))
    }

    pub fn cbrt(self) -> Node {
        Node::Exp(Box::new(self), Box::new(common::three().inverse()))
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
        common::minus_one() * self
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

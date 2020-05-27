use float_cmp::{ApproxEq, F64Margin};
use num_traits::Float;
use std::f64::consts::{E, PI};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::*;

use super::util::{fold_nodes, get_op_result_base};
use super::{ConstKind, Node};
use crate::ratio2flt::ratio_to_f64;

/// A struct that holds the result of a calculation.
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct EvalSuccess {
    /// The result value
    pub val: f64,

    /// The base the result value should be displayed in
    pub display_base: Option<u32>,
}

/// A description of the error of a calculation.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum EvalError {
    ZeroToThePowerOfZero,
    Tan90Or270,
}

/// Approximates the node value.
pub fn eval(node: &Node) -> Result<EvalSuccess, EvalError> {
    Ok(match node {
        Node::Const(kind) => EvalSuccess {
            val: match kind {
                ConstKind::Pi => PI,
                ConstKind::Tau => PI * 2.0,
                ConstKind::E => E,
            },
            display_base: None,
        },
        Node::Num { val, input_base } => EvalSuccess {
            val: ratio_to_f64(&val),
            display_base: *input_base,
        },
        Node::Sum(children) => fold_nodes(children.iter(), 0.0, Add::add)?,
        Node::Product(children) => fold_nodes(children.iter(), 1.0, Mul::mul)?,
        Node::Exp(a, b) => {
            let a = eval(a)?;
            let b = eval(b)?;
            if b.val.approx_eq(
                0.0,
                F64Margin {
                    epsilon: 0.0,
                    ulps: 2,
                },
            ) {
                if a.val.approx_eq(
                    0.0,
                    F64Margin {
                        epsilon: 0.0,
                        ulps: 2,
                    },
                ) {
                    return Err(EvalError::ZeroToThePowerOfZero);
                }
                return Ok(EvalSuccess {
                    val: 1.0,
                    display_base: a.display_base,
                });
            } else if b.val.approx_eq(
                1.0,
                F64Margin {
                    epsilon: 0.0,
                    ulps: 3,
                },
            ) {
                return Ok(EvalSuccess {
                    val: a.val,
                    display_base: a.display_base,
                });
            } else if b.val.approx_eq(
                -1.0,
                F64Margin {
                    epsilon: 0.0,
                    ulps: 3,
                },
            ) {
                return Ok(EvalSuccess {
                    val: 1.0 / a.val,
                    display_base: a.display_base,
                });
            }
            EvalSuccess {
                val: a.val.powf(b.val),
                display_base: get_op_result_base(a.display_base, b.display_base),
            }
        }
        Node::Sin(inner) => eval_map(inner, f64::sin, false)?,
        Node::Cos(inner) => eval_map(inner, f64::cos, false)?,
        Node::Tan(inner) => {
            let original = eval(inner)?;
            if original.val.is_infinite() {
                // don't even try anymore
                return Ok(EvalSuccess {
                    val: 0.0,
                    display_base: None,
                });
            }
            assert!(!original.val.is_nan());
            // check if angle is k(2pi) + pi/2 or k(2pi) + 3*pi/2
            let n = (original.val / (PI * 2.0)).floor();
            let impossible1 = n * (PI * 2.0) + PI / 2.0;
            let impossible2 = n * (PI * 2.0) + 3.0 * PI / 2.0;
            if original.val.approx_eq(
                impossible1,
                F64Margin {
                    epsilon: 0.0,
                    ulps: 3,
                },
            ) || original.val.approx_eq(
                impossible2,
                F64Margin {
                    epsilon: 0.0,
                    ulps: 3,
                },
            ) {
                return Err(EvalError::Tan90Or270);
            }
            EvalSuccess {
                val: original.val.tan(),
                display_base: None,
            }
        }
    })
}

fn eval_map<F: Fn(f64) -> f64>(
    node: &Node,
    f: F,
    keep_base: bool,
) -> Result<EvalSuccess, EvalError> {
    let original = eval(node)?;
    if original.val.is_infinite() {
        // don't even try anymore
        return Ok(EvalSuccess {
            val: 0.0,
            display_base: None,
        });
    }
    assert!(!original.val.is_nan());
    Ok(EvalSuccess {
        val: f(original.val),
        display_base: if keep_base {
            original.display_base
        } else {
            None
        },
    })
}

impl Display for EvalSuccess {
    fn fmt(&self, out: &mut Formatter) -> fmt::Result {
        match self.display_base.unwrap_or(10) {
            10 => self.val.fmt(out),

            // TODO: add tests
            2 => {
                let (mantissa, exp, sign) = self.val.integer_decode();
                let is_nonnegative = sign == 1 || mantissa == 0;
                let mut result = String::new();
                let mut is_printing = true;
                let mut consecutive_zeros = 0;

                if exp <= -53 {
                    result += "0.";
                    for _ in exp..-53 {
                        result.push('0');
                    }
                }
                for i in (0..53).rev() {
                    let bit = mantissa & (1 << i) != 0;
                    let bit_exp = exp + i;
                    if bit_exp < 0 {
                        // check how many digits we must print after the decimal separator
                        if let Some(precision) = out.precision() {
                            if (-bit_exp as usize) > precision {
                                break;
                            }
                        }
                    }
                    // do not print leading zeros
                    if bit || bit_exp < 0 {
                        is_printing = true;
                    }
                    if is_printing {
                        if bit_exp < 0 && !bit && out.precision().is_none() {
                            // Do not print zeros after the decimal separator
                            // unless there is a one after.
                            // Example: 0b1.100000000 -> 0b1.1
                            consecutive_zeros += 1;
                        } else {
                            if consecutive_zeros > 0 {
                                // we forgot to print the decimal separator
                                if bit_exp + consecutive_zeros == -1 {
                                    result.push('.');
                                }
                                for _ in 0..consecutive_zeros {
                                    result.push('0');
                                }
                                consecutive_zeros = 0;
                            }
                            if i != 52 && bit_exp == -1 {
                                // we got our first digit after a dot!
                                result.push('.');
                            }
                            result.push(if bit { '0' } else { '1' });
                        }
                    }
                }
                if let Some(precision) = out.precision() {
                    if exp < 0 {
                        let already_printed = -exp as usize;

                        // fill with zeros at the end
                        for _ in already_printed..precision {
                            result.push('0');
                        }
                    }
                }

                out.pad_integral(is_nonnegative, "0b", &result)
            }
            _ => {
                eprintln!("cannot print result in bases other than 10 and two yet");
                self.val.fmt(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use num_rational::BigRational;

    use super::*;
    use crate::node::util::common;

    #[test]
    fn it_errors_with_pow_0_0() {
        // 0^0
        let input = Node::Exp(Box::new(common::zero()), Box::new(common::zero()));
        let result = eval(&input);
        assert_eq!(result, Err(EvalError::ZeroToThePowerOfZero));
    }

    #[test]
    fn it_errors_with_impossible_tangent_inside_0_2pi() {
        // tan(pi/2)
        let input = Node::Tan(Box::new(Node::Const(ConstKind::Pi) / common::two()));
        let result = eval(&input);
        assert_eq!(result, Err(EvalError::Tan90Or270));
    }

    #[test]
    fn it_errors_with_impossible_tangent_outside_0_2pi() {
        // tan(97pi/2)
        let ninety_seven = Node::Num {
            val: BigRational::from_integer(97.into()),
            input_base: Some(10),
        };
        let input = Node::Tan(Box::new(
            ninety_seven * Node::Const(ConstKind::Pi) / common::two(),
        ));
        let result = eval(&input);
        assert_eq!(result, Err(EvalError::Tan90Or270));
    }
}

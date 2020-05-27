use std::fmt;
use std::fmt::{Display, Formatter};

use num_traits::Float;

/// A struct that holds the result of a calculation.
pub struct EvalResult {
    /// The result value
    pub val: f64,

    /// The base the result value should be displayed in
    pub display_base: Option<u32>,
}

impl Display for EvalResult {
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

            _ => todo!(),
        }
    }
}

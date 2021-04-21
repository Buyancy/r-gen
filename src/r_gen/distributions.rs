//A module that will handle the distributions for the library. 

use std::{fmt, ops::{Add, Div, Mul, Sub}};

use probability::prelude::*;
use rand::{self, FromEntropy, prelude::ThreadRng, rngs::StdRng}; 
use rand::distributions::Distribution as Distr;




//A value struct that will handle possible values from the distributions. 
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Boolean(bool), 
    Integer(i64), 
    Real(f64)
}

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Boolean(b) => {
                write!(formatter, "{}", b)
            }, 
            Value::Integer(i) => {
                write!(formatter, "{}", i)
            }, 
            Value::Real(r) => {
                write!(formatter, "{}", r)
            },
        }
    }
}

//Implement type conversions
impl Into<f64> for Value {
    fn into(self) -> f64 { 
        match self {
            Value::Real(v) => v, 
            _ => panic!("Tried to convert non Real value into a f64.")
        }
    }
}
impl Into<Value> for f64 {
    fn into(self) -> Value { 
        Value::Real(self)
    }
}
impl Into<i64> for Value {
    fn into(self) -> i64 { 
        match self {
            Value::Integer(v) => v, 
            _ => panic!("Tried to convert non Integer value into an i64.")
        }
    }
}
impl Into<Value> for i64 {
    fn into(self) -> Value { 
        Value::Integer(self)
    }
}
impl Into<bool> for Value {
    fn into(self) -> bool { 
        match self {
            Value::Boolean(v) => v, 
            _ => panic!("Tried to convert non Boolean value into a bool.")
        }
    }
}
impl Into<Value> for bool {
    fn into(self) -> Value { 
        Value::Boolean(self)
    }
}

//Implement mathematical operations for the functions.
impl Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        match self {
            Self::Integer(i1) => {
                match rhs {
                    Self::Integer(i2) => Self::Integer(i1+i2), 
                    Self::Real(r2) => Self::Real(i1 as f64 + r2), 
                    Self::Boolean(_) => panic!("Unable to add boolean values.")
                }
            }, 
            Self::Real(r1) => {
                match rhs {
                    Self::Integer(i2) => Self::Real(r1 + i2 as f64), 
                    Self::Real(r2) => Self::Real(r1 + r2), 
                    Self::Boolean(_) => panic!("Unable to add boolean values.")
                }
            }, 
            Self::Boolean(_) => panic!("Unable to add boolean values."),
        }
    }
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        match self {
            Self::Integer(i1) => {
                match rhs {
                    Self::Integer(i2) => Self::Integer(i1 - i2), 
                    Self::Real(r2) => Self::Real(i1 as f64 - r2), 
                    Self::Boolean(_) => panic!("Unable to subtract boolean values.")
                }
            }, 
            Self::Real(r1) => {
                match rhs {
                    Self::Integer(i2) => Self::Real(r1 - i2 as f64), 
                    Self::Real(r2) => Self::Real(r1 - r2), 
                    Self::Boolean(_) => panic!("Unable to subtract boolean values.")
                }
            }, 
            Self::Boolean(_) => panic!("Unable to subtract boolean values."),
        }
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        match self {
            Self::Integer(i1) => {
                match rhs {
                    Self::Integer(i2) => Self::Integer(i1 * i2), 
                    Self::Real(r2) => Self::Real(i1 as f64 * r2), 
                    Self::Boolean(_) => panic!("Unable to multiply boolean values.")
                }
            }, 
            Self::Real(r1) => {
                match rhs {
                    Self::Integer(i2) => Self::Real(r1 * i2 as f64), 
                    Self::Real(r2) => Self::Real(r1 * r2), 
                    Self::Boolean(_) => panic!("Unable to multiply boolean values.")
                }
            }, 
            Self::Boolean(_) => panic!("Unable to multiply boolean values."),
        }
    }
}

impl Div<Value> for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Self::Output {
        match self {
            Self::Integer(i1) => {
                match rhs {
                    Self::Integer(i2) => Self::Integer(i1 / i2), 
                    Self::Real(r2) => Self::Real(i1 as f64 / r2), 
                    Self::Boolean(_) => panic!("Unable to divide boolean values.")
                }
            }, 
            Self::Real(r1) => {
                match rhs {
                    Self::Integer(i2) => Self::Real(r1 / i2 as f64), 
                    Self::Real(r2) => Self::Real(r1 / r2), 
                    Self::Boolean(_) => panic!("Unable to divide boolean values.")
                }
            }, 
            Self::Boolean(_) => panic!("Unable to divide boolean values."),
        }
    }
}





//The distributions we can use. 
#[derive(Debug)]
pub enum Distribution {
    Bernoulli(f64), 
    Binomial(i64, f64), 
    Normal(f64, f64), 
    Gamma(f64, f64), 
    Beta(f64, f64)
}

pub struct Source<T>(pub T);

impl<T: rand::RngCore> source::Source for Source<T> {
    fn read_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
}

impl Distribution {
    //Sample from the distribution. 
    pub fn sample(&self) -> Value {
        match self {
            Distribution::Bernoulli(p) => {
                let d = rand::distributions::Bernoulli::new(*p);
                let v = d.sample(&mut ThreadRng::default());
                Value::Boolean(v)
            }, 
            Distribution::Binomial(n, p) => {
                let b = probability::distribution::Binomial::new(*n as usize, *p); 
                Value::Integer(b.sample(&mut Source(StdRng::from_entropy())) as i64)
            }, 
            Distribution::Normal(mu, sigma_squared) => {
                let n = probability::distribution::Gaussian::new(*mu, *sigma_squared); 
                Value::Real(n.sample(&mut Source(StdRng::from_entropy())))
            }, 
            Distribution::Gamma(alpha, beta) => {
                let g = probability::distribution::Gamma::new(*alpha, *beta); 
                Value::Real(g.sample(&mut Source(StdRng::from_entropy())))
            }, 
            Distribution::Beta(alpha, beta) => {
                let b = probability::distribution::Beta::new(*alpha, *beta, 0.0, 1.0);
                Value::Real(b.sample(&mut Source(StdRng::from_entropy()))) 
            }
        }
    }

    //Compute the liklihood of a value given a distribution. 
    pub fn liklihood(&self, value : &Value) -> Result<f64, &str> {
        match self {
            Distribution::Bernoulli(p) => {
                match value {
                    Value::Boolean(b) => {
                        match b {
                            true  => Ok(p.ln()), 
                            false => Ok((1.0 - p).ln()), 
                        }
                    }, 
                    _ => Err("Value of wrong type, expected Boolean.")
                }
            }, 
            Distribution::Binomial(n, p) => {
                match value {
                    Value::Integer(k) => {
                        let norm = probability::distribution::Binomial::new(*n as usize, *p);
                        Ok(norm.mass(*k as usize).ln())
                    }, 
                    _ => Err("Value of wrong type, expected Integer.")
                }
            }, 
            Distribution::Normal(mu, sigma_squared) => {
                match value {
                    Value::Real(n) => {
                        let norm = probability::distribution::Gaussian::new(*mu, *sigma_squared); 
                        Ok(norm.density(*n).ln())
                    }, 
                    _ => Err("Value of wrong type, expected Real.")
                }
            },
            Distribution::Gamma(alpha, beta) => {
                match value {
                    Value::Real(n) => {
                        let g = probability::distribution::Gaussian::new(*alpha, *beta); 
                        Ok(g.density(*n).ln())
                    }, 
                    _ => Err("Value of wrong type, expected Real.")
                }
            }, 
            Distribution::Beta(alpha, beta) => {
                match value {
                    Value::Real(n) => {
                        let b = probability::distribution::Beta::new(*alpha, *beta, 0.0, 1.0);
                        Ok(b.density(*n).ln())
                    }, 
                    _ => Err("Value of wrong type, expected Real.")
                }
            }
        }
    }
}
//! Distributions a generative model can sample from. 

use std::{fmt, ops::{Add, Div, Mul, Sub}};

use probability::prelude::*;
use rand::{self, FromEntropy, prelude::ThreadRng, rngs::StdRng}; 
use rand::distributions::Distribution as Distr;

/**
A value struct that will handle possible values from the distributions.
*/ 
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Represents a boolean. 
    Boolean(bool), 
    /// Represents an integer. 
    Integer(i64), 
    /// Represents a real number. 
    Real(f64), 
    /// Represents a vector.
    Vector(Vec<Value>)
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
            Value::Vector(v) => {
                write!(formatter, "{:?}", v)
            }, 
        }
    }
}

//Implement type conversions
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Boolean(b)
    }
}
impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Real(f)
    }
}
impl From<f32> for Value {
    fn from(f: f32) -> Self {
        Value::Real(f as f64)
    }
}
impl From<i8> for Value {
    fn from(i: i8) -> Self {
        Value::Integer(i as i64)
    }
}
impl From<i16> for Value {
    fn from(i: i16) -> Self {
        Value::Integer(i as i64)
    }
}
impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Integer(i as i64)
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Integer(i)
    }
}
impl From<u8> for Value {
    fn from(i: u8) -> Self {
        Value::Integer(i as i64)
    }
}
impl From<u16> for Value {
    fn from(i: u16) -> Self {
        Value::Integer(i as i64)
    }
}
impl From<u32> for Value {
    fn from(i: u32) -> Self {
        Value::Integer(i as i64)
    }
}
impl From<u64> for Value {
    fn from(i: u64) -> Self {
        Value::Integer(i as i64)
    }
}
impl From<usize> for Value {
    fn from(i: usize) -> Self {
        Value::Integer(i as i64)
    }
}

impl Into<f64> for Value {
    fn into(self) -> f64 {
        match self {
            Value::Real(r) => r, 
            _ => panic!("Cannot convert non-Real to f64.")
        }
    }
}
impl Into<i64> for Value {
    fn into(self) -> i64 {
        match self {
            Value::Integer(i) => i, 
            _ => panic!("Cannot convert non-Integer to i64.")
        }
    }
}
impl Into<bool> for Value {
    fn into(self) -> bool {
        match self {
            Value::Boolean(b) => b, 
            _ => panic!("Cannot convert non-Boolean to bool.")
        }
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
                    Self::Boolean(_) => panic!("Unable to add boolean values."), 
                    Self::Vector(_) => panic!("Ubalbe to add integer to vector.")
                }
            }, 
            Self::Real(r1) => {
                match rhs {
                    Self::Integer(i2) => Self::Real(r1 + i2 as f64), 
                    Self::Real(r2) => Self::Real(r1 + r2), 
                    Self::Boolean(_) => panic!("Unable to add boolean values."), 
                    Self::Vector(_) => panic!("Unable to add Real value to Vector")
                }
            }, 
            Self::Boolean(_) => panic!("Unable to add boolean values."),
            Self::Vector(vl) => {
                match rhs {
                    Self::Vector(vr) => Self::Vector(vl.iter().zip(vr.iter()).map(|(l, r)| l.clone() + r.clone()).collect()), 
                    _ => panic!("Unable to add Vector to non-Vector.")
                }
            }
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
                    Self::Boolean(_) => panic!("Unable to subtract boolean values."), 
                    Self::Vector(_) => panic!("Ubalbe to subtract vector form integer.")
                }
            }, 
            Self::Real(r1) => {
                match rhs {
                    Self::Integer(i2) => Self::Real(r1 - i2 as f64), 
                    Self::Real(r2) => Self::Real(r1 - r2), 
                    Self::Boolean(_) => panic!("Unable to subtract boolean values."),
                    Self::Vector(_) => panic!("Unalbe to subtract Real from Vector.")
                }
            }, 
            Self::Boolean(_) => panic!("Unable to subtract boolean values."),
            Self::Vector(vl) => {
                match rhs {
                    Self::Vector(vr) => Self::Vector(vl.iter().zip(vr.iter()).map(|(l, r)| l.clone() - r.clone()).collect()), 
                    _ => panic!("Unable to subtract non-Vector from Vector.")
                }
            }
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
                    Self::Boolean(_) => panic!("Unable to multiply boolean values."), 
                    Self::Vector(_) => panic!("Unable to multiply vectors.")
                }
            }, 
            Self::Real(r1) => {
                match rhs {
                    Self::Integer(i2) => Self::Real(r1 * i2 as f64), 
                    Self::Real(r2) => Self::Real(r1 * r2), 
                    Self::Boolean(_) => panic!("Unable to multiply boolean values."),
                    Self::Vector(_) => panic!("Unable to multiply vectors.")
                }
            }, 
            Self::Boolean(_) => panic!("Unable to multiply boolean values."),
            Self::Vector(_) => panic!("Unable to multiply vectors.")
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
                    Self::Boolean(_) => panic!("Unable to divide boolean values."),
                    Self::Vector(_) => panic!("Unable to divide vectors.")
                }
            }, 
            Self::Real(r1) => {
                match rhs {
                    Self::Integer(i2) => Self::Real(r1 / i2 as f64), 
                    Self::Real(r2) => Self::Real(r1 / r2), 
                    Self::Boolean(_) => panic!("Unable to divide boolean values."),
                    Self::Vector(_) => panic!("Unable to divide vectors.")
                }
            }, 
            Self::Boolean(_) => panic!("Unable to divide boolean values."),
            Self::Vector(_) => panic!("Unable to divide vectors.")
        }
    }
}





/// Enum that uniquely discribes a given distribution. 
#[derive(Debug)]
pub enum Distribution {
    /// A Bernoulli distribution with paramater p.
    Bernoulli(f64),     
    /// A Binomial distribution with paramaters n and p.    
    Binomial(i64, f64),     
    /// A Normal distribution with paramaters mu and sigma.
    Normal(f64, f64),       
    /// A Gamma distribution with parameters alpha and beta.
    Gamma(f64, f64),        
    /// A Beta distribution with parameters alpha and beta.
    Beta(f64, f64),         
    /// A Lognormal distribution with paramaters mu and sigma.
    LogNormal(f64, f64), 
    /// A Categorical distribution with weights equal to p.
    Categorical(Vec<f64>), 
    /// A Dirichlet distribution that returns a vector of degree n. 
    Dirichlet(Vec<f64>),
}

/**
A trait that you can implement to create your own distributions to sample from in a 
generative model. 
*/
pub trait Sampleable {
    /// Sample a value from the distribution. 
    fn sample(&self) -> Value; 
    /// Compute the liklihood of a given value being sampled from the distribution.
    fn liklihood(&self, value : &Value) -> Result<f64, &str>; 
}

/// A struct that holds a source of randomness for the various distributions.
pub(crate) struct Source<T>(pub T);
impl<T: rand::RngCore> source::Source for Source<T> {
    fn read_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
}

impl Sampleable for Distribution {
    /// Sample from the distribution and return the value sampled. 
    fn sample(&self) -> Value {
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
            }, 
            Distribution::LogNormal(mu, sigma) => {
                let n = probability::distribution::Lognormal::new(*mu, *sigma); 
                Value::Real(n.sample(&mut Source(StdRng::from_entropy())))
            }, 
            Distribution::Categorical(v) => {
                let c = probability::distribution::Categorical::new(&v[..]); 
                Value::Integer(c.sample(&mut Source(StdRng::from_entropy())) as i64)
            }, 
            Distribution::Dirichlet(xs) => {
                let ys : Vec<f64> = xs.iter().map(|x|{ 
                    let beta = probability::distribution::Beta::new(*x, 1.0, 0.0, 1.0); 
                    beta.sample(&mut Source(StdRng::from_entropy()))
                }).collect(); 
                let sum : f64 = ys.iter().sum(); 
                let ys = ys.iter().map(|y| Value::Real(y / sum)).collect(); 
                Value::Vector(ys)
            }
        }
    }

    /**
    Compute the liklihood of a value given a distribution (returns the log liklihood.) 
    # Errors 
    This function will return an Err if you try to determine the liklihood of a variant of the ```Value``` enum that 
    the distribution does not produce. For example, trying to get the liklihood of a real number from a bernoulli 
    distribution will return an Err.
    */
    fn liklihood(&self, value : &Value) -> Result<f64, &str> {
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
            }, 
            Distribution::LogNormal(mu, sigma) => {
                match value {
                    Value::Real(n) => {
                        let l = probability::distribution::Lognormal::new(*mu, *sigma); 
                        Ok(l.density(*n).ln())
                    }, 
                    _ => Err("Value of wrong type, expected Real.")
                }
            }, 
            Distribution::Categorical(p) => {
                match value {
                    Value::Integer(i) => {
                        let c = probability::distribution::Categorical::new(&p[..]); 
                        Ok(c.mass(*i as usize).ln())
                    }, 
                    _ => Err("Value of wrong type, expected Integer.")
                }
            }, 
            Distribution::Dirichlet(a) => {
                match value {
                    Value::Vector(x) => {
                        let ba_numerator = a.iter().fold(1.0, |acc, x| acc * gamma(*x)); 
                        let ba_denominator : f64 = gamma(a.iter().sum());  
                        let ba = ba_numerator / ba_denominator; 

                        Ok((1.0/ba) * x.iter().map(|x| {
                            match *x {
                                Value::Real(x) => x, 
                                _ => 1.0, 
                            }
                        }).zip(a.iter()).fold(1.0, |acc, (x, a)| {
                            acc * x.powf(a-1.0)
                        }))
                    }, 
                    _ => Err("Value of wrong type, expected Vector.")
                }
            }
        }
    }
}

//A helper method that computes the gamma function. 
const TAYLOR_COEFFICIENTS: [f64; 29] = [
    -0.00000000000000000023,  0.00000000000000000141,  0.00000000000000000119,
    -0.00000000000000011813,  0.00000000000000122678, -0.00000000000000534812,
    -0.00000000000002058326,  0.00000000000051003703, -0.00000000000369680562,
     0.00000000000778226344,  0.00000000010434267117, -0.00000000118127457049,
     0.00000000500200764447,  0.00000000611609510448, -0.00000020563384169776,
     0.00000113302723198170, -0.00000125049348214267, -0.00002013485478078824,
     0.00012805028238811619, -0.00021524167411495097, -0.00116516759185906511,
     0.00721894324666309954, -0.00962197152787697356, -0.04219773455554433675,
     0.16653861138229148950, -0.04200263503409523553, -0.65587807152025388108,
     0.57721566490153286061,  1.00000000000000000000,
];
const INITIAL_SUM: f64 = 0.00000000000000000002;
fn gamma(x: f64) -> f64 {
    TAYLOR_COEFFICIENTS.iter().fold(INITIAL_SUM, |sum, coefficient| {
        sum * (x - 1.0) + coefficient
    }).recip()
}
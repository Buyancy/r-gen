# r-gen
A generative programming framework for the rust programming language. This is heavily based on the Gen library for the Julia programming language.

Example using importance resampling to estimate latent variables.
```rust
...
fn main() {
    //Define our generative model. 
    #[r_gen]
    fn my_model(():()){
        sample!(p ~ Distribution::Beta(1.0, 1.0)); 
        sample!(num_heads ~ Distribution::Binomial(100, p.into()));
    }

    //Run the model once in the forward direction and record the observations. 
    let (t, _) : (Trace, _)= simulate(&mut my_model, ());
    let choices = Choicemap::from(vec![("num_heads", t.choices["num_heads"])]);

    //Perform importance resampling to get an estimate for the value of p. 
    let mut traces = Vec::new();
    for _ in 0..1000 {
        let (gt, _) : (Trace, _)= generate(&mut my_model, (), &choices);
        traces.push(gt); 
    }
    
    println!("Actual value for p:\t {}", t.choices["p"]); 
    println!("Generated value for p:\t {}", Trace::sample_weighted_traces(&traces).unwrap().choices["p"]); 
}
```
Outputs:
```
Actual value for p:      0.8011431168181488
Generated value for p:   0.7879998086169554
```

Note the `#[r_gen]` tag on the generative function. This must be on any of your generative functions in order to use `simulate` and `generate` on them. In order to make a random choice (sample from a distribution) you must use the `sample!()` macro. The syntax for this is 
```rust
sample!(identifier ~ Distribution); 
```
This samples the `Distribution` given and stores the result in `identifier`. `Distribution` is defined as 
```rust
pub enum Distribution {
    Bernoulli(f64),         //p
    Binomial(i64, f64),     //n, p
    Normal(f64, f64),       //mu, sigma
    Gamma(f64, f64),        //alpha, beta
    Beta(f64, f64),         //alpha, beta
    LogNormal(f64, f64),    //mu, sigma 
    Categorical(Vec<f64>),  //p
    Dirichlet(Vec<f64>),    //alpha
}
```
`Value` is defined as 
```rust 
pub enum Value {
    Boolean(bool), 
    Integer(i64), 
    Real(f64), 
    Vector(Vec<Value>)
}
```
The variant of the enum that is returned from sampling depends on the distribution being sampled.

---

This framework is in EARLY stages of development. Everything is subject to change.  

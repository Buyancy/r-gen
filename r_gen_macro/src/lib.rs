use proc_macro::{TokenStream, TokenTree};
use proc_macro::TokenTree::*; 
use core::iter::FromIterator; 
use core::str::FromStr; 
use proc_macro::{Group}; 
use proc_macro::Delimiter; 
use itertools::izip;


//The macro to convert the function signature. 
#[proc_macro_attribute]
pub fn r_gen(_attr: TokenStream, item: TokenStream) -> TokenStream {

    let mut it = item.clone().into_iter();
    let mut out : Vec<TokenTree> = Vec::new();  

    //Verify that this is a function. 
    if let Some(Ident(i)) = it.next() {
        //We are dealing with a function. 
        out.push(Ident(i)); 
        //println!("FUNCTION")
    } else {
        panic!("The #[r_gen] macro can only be applied to functions.") 
    }

    //Get the name of the function. 
    if let Some(Ident(name)) = it.next() {
        //println!("NAME: {:}", &name); 
        out.push(Ident(name)); 
        
    } else {
        panic!("Generative functions require a name.")
    }

    //Get the parameters of the function. 
    if let Some(Group(args)) = it.next() {
        //println!("OLD FUNCTION ARGS: {:}", &args);
        let new_args = get_new_args(args); 
        //println!("NEW FUNCTION ARGS: {:}", &new_args);
        out.push(Group(new_args));
    } else {
        panic!("Malformed generative function. Could not identify function arguments.")
    }

    //Handle the body. 
    if let Some(Group(body)) = it.next() {
        out.push(TokenTree::Group(update_body(body)))
    }
     
    let out = TokenStream::from_iter(out.into_iter());

    out

}

//A function that will take the old arguments and spit out the new arguments. 
/*
    NOTE: 
    This is working for now as long as there is only a single argument (it can be a tuple.)
    In the future, I want to take multiple arguments and convert them into a tuple for the 
    user before the final argument is added to the converted function signature. 
*/
fn get_new_args(old_args : Group) -> Group {
    //Start by parsing the sample function.
    let mut samp_trace_arg = TokenStream::from_str("mut _sample : Rc<dyn FnMut(&String, Distribution, &mut Trace) -> Value>, _trace : &mut Trace, ").unwrap();
    let new_args = old_args; //Need to handle multiple args here.
    samp_trace_arg.extend(new_args.stream());
    let new_args = Group::new(Delimiter::Parenthesis, samp_trace_arg); 
    new_args
}

//A function that will add _sample and _trace identifiers to the sample! macros in the body. 
//This function will make recursive calls on sub-scopes. 
fn update_body(body : Group) -> Group {
    let g = Group::new(Delimiter::Brace, update_tok_stream(body.stream())); 
    g
    //Group::new(Delimiter::Brace, TokenStream::new())
}

fn update_tok_stream(tok_stream : TokenStream) -> TokenStream {
    let mut res = TokenStream::new(); 
    let tracking_stream = 
        izip!(
            tok_stream.clone().into_iter(), 
            tok_stream.clone().into_iter().skip(1), 
            tok_stream.clone().into_iter().skip(2)); 

    let mut ti = tok_stream.clone().into_iter(); 

    if let Some(t) = ti.next() {
        res.extend(TokenStream::from(t)); 
    } else {
        return tok_stream; 
    }

    if let Some(t) = ti.next() {
        res.extend(TokenStream::from(t)); 
    } else {
        return tok_stream; 
    }

    for (prev_prev, prev, tok) in tracking_stream {
        match &tok {
            Group(g) => {
                match (prev, prev_prev) {
                    (Punct(p), Ident(i)) => {
                        if p.as_char() == '!' && i.to_string() == "sample" {
                            res.extend(update_sample_params(g.clone())); 
                            //println!("FOUND SAMPLE: {:?} {:?} {}", p.as_char(), i.to_string(), &tok); 
                        } else {
                            res.extend(TokenStream::from(tok));
                        }
                    }, 
                    _ => {
                        res.extend(TokenStream::from(TokenTree::Group(Group::new(g.delimiter(), update_tok_stream(g.stream())))));
                    }
                }
            }
            _ => {
                res.extend(TokenStream::from(tok)); 
            } 
        }
    }
    res
}

fn update_sample_params(group : Group) -> TokenStream {
    let mut new_params = TokenStream::from_str("_sample _trace ").unwrap();
    new_params.extend(group.stream()); 
    TokenStream::from(TokenTree::Group(Group::new(Delimiter::Parenthesis, new_params)))
}


/* 
Want to take this: 

    #[r_gen(attr)]
    fn flip_biased_coin(p : f64) -> f64 {
        (flip, i) ~ Bernoulli(p);

        print!(flip);
        flip
    }

And convert it to this: 

    fn flip_biased_coin(mut __sample : Rc<dyn FnMut(&String, Distribution, &mut Trace) -> ()>, __trace : &mut Trace, p : f64) {
        flip = (Rc::get_mut(&mut __sample).unwrap())(&String::from("flip"), Distribution::Bernoulli(p), __trace);
        print!(flip); 
        flip
    }

*/
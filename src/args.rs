
mod kyopro_args {
    use regex::Regex;
    use itertools::Itertools;

    pub struct Args {
        args_str: String,
    }

    impl Args {
        pub fn new() -> Self {
            let args: Vec<String> = std::env::args().collect();
            let args_str = args[1..].iter().join(" ");
            Self { args_str }
        }

        pub fn get_num<T: std::str::FromStr>(&self, arg_name: &str) -> Option<T> {
            let re_str = format!(r"-{}=([\d.-]+)", arg_name);
            let re = Regex::new(&re_str).unwrap();
            let Some(captures) = re.captures(&self.args_str) else { return None; };
            captures[1].parse().ok()
        }
    }
}

use kyopro_args::Args;

fn main() {
    let args = Args::new();
    let x: usize = args.get_num::<usize>("x").unwrap();
    let y: f64 = args.get_num::<f64>("y").unwrap();
    println!("x: {}, y: {}", x, y);
}


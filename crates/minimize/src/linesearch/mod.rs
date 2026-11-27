//! Moré–Thuente strong-Wolfe line search.

mod config;
mod dcstep;
mod driver;
mod more_thuente;

pub use config::{ConfigError, MoreThuenteConfig};
pub use driver::{scalar_search_wolfe1, Wolfe1Error, Wolfe1Options, Wolfe1Outcome, Wolfe1Report};
pub use more_thuente::{MoreThuente, Point, Reason, Stage, StartError, Step, Termination};

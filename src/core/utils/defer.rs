//! Scope-exit guards for cleanup and temporary state changes.
//!
//! The macros create local drop guards that run as control leaves the
//! surrounding scope, whether that
//! is a normal return, an early `?`, or an unwind. That last case is the one
//! that matters at the module boundary: a guard still runs when a panic
//! crosses it.

/// Runs an action when the surrounding scope exits.
///
/// Multiple guards in one scope run in reverse declaration order.
///
/// ```ignore
/// defer! {{ tracing::debug!("left the frame"); }}
/// ```
#[macro_export]
macro_rules! defer {
	($body:block) => {
		struct _Defer_<F: FnMut()> {
			closure: F,
		}

		impl<F: FnMut()> Drop for _Defer_<F> {
			fn drop(&mut self) { (self.closure)(); }
		}

		let _defer_ = _Defer_ { closure: || $body };
	};

	($body:expr_2021) => {
		$crate::defer! {{ $body }}
	};
}

/// Temporarily replaces a value and restores the previous one at scope exit.
///
/// The first argument names a mutable reference; the second becomes its value
/// until the scope ends. Restoration happens on unwind too.
#[macro_export]
macro_rules! scope_restore {
	($val:ident, $ours:expr_2021) => {
		let theirs = $crate::utils::exchange($val, $ours);
		$crate::defer! {{ *$val = theirs; }};
	};
}

#[cfg(test)]
mod tests {
	use std::{
		cell::Cell,
		panic::{AssertUnwindSafe, catch_unwind},
	};

	#[test]
	fn a_guard_runs_at_the_end_of_its_scope() {
		let ran = Cell::new(false);
		let seen_inside = {
			crate::defer! {{ ran.set(true); }}

			ran.get()
		};

		assert!(!seen_inside, "not before the scope ends");
		assert!(ran.get(), "the guard runs on the way out");
	}

	#[test]
	fn a_guard_runs_while_unwinding() {
		let ran = Cell::new(false);
		let result = catch_unwind(AssertUnwindSafe(|| {
			crate::defer! {{ ran.set(true); }}
			panic!("on the way through");
		}));

		assert!(result.is_err(), "the panic is caught here, not swallowed");
		assert!(ran.get(), "a panic crossing the scope still runs the guard");
	}

	#[test]
	fn a_replaced_value_comes_back() {
		// @note: the value cannot be read inside the scope. The restore guard
		// holds the mutable borrow until the scope ends, which is exactly the
		// property that makes the restore unconditional.
		let mut held = 1;
		{
			let value = &mut held;
			crate::scope_restore!(value, 2);
		}

		assert_eq!(held, 1, "theirs again afterwards");
	}

	#[test]
	fn exchange_hands_back_what_was_there() {
		let mut held = 1;
		let previous = crate::utils::exchange(&mut held, 2);

		assert_eq!(previous, 1, "the caller gets the old value to put back later");
		assert_eq!(held, 2, "and the slot holds the new one");
	}
}

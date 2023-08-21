# Release v0.1.4

In Julia v1.10, the package would not be able to precompile due to struct constructors and add_event method definitions being redefined in multiple lines. Previously in v1.9, Julia would just flag an error and allow you to still use the package, in v1.10, Julia would just not let you use the package.

This release allows the package to precompile properly while being able to pass all the tests that have been defined. At the expense of some extra lines of code, we tried to keep to clarity in the different method/struct argument combinations that are possible. This often involved removing default arguments in positional argument functions and defining separate functions that took fewer arguments.

# Release v0.1.3

https://github.com/JuliaRegistries/General/pull/54715

- Fixed multiplayer evidence
- A performance test between the Distributions package and own solution.
- Replace `!isdisjoint(a,b)` by `lenght(intersect(a,b))>0`, because `isdisjoint` requires julia 1.5

# Release v0.1.2

- The History class 
- Docs

## Visualizing Hurricane Data

Inspired by:
[/r/dataisbeautiful and /u/Tjukanov](https://www.reddit.com/r/dataisbeautiful/comments/6y0h2q/100_years_of_hurricane_paths_animated_oc/)

That ended up being this kind of thought process:

* Ignored all storm basins except the North American Atlantic Basin as that's where most of the (Western) damage is concentrated.  (However Asian typhoons have been measured to be quite a bit stronger.)
* I took each track and figured out the category of the hurricane at any given time. 
* At each time interval apply some 'damage' to the lat/long that the storm is currently located at.
* The damage formula was: (storm category + 2)^2   This accounted for tropical storms and tropical depressions. The square is because the Saffir-Simpson Hurricane Categories are a roughly logarithmic scale.  Also, according to the Saffir-Simpson scale, wind is the primary source of damage from hurricanes and air pressure increases as the square of velocity.

That lead to this gif:
[100 Years of Hurricane Intensities](https://gfycat.com/SecondhandShrillHamster)

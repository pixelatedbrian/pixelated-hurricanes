## Visualizing Hurricane Data

The documentation of this repo is currently in a shambles but some effort will be made soon to make things more clearly explained in terms of what script generates what kind of video. -brian

Inspired by:
[/r/dataisbeautiful and /u/Tjukanov](https://www.reddit.com/r/dataisbeautiful/comments/6y0h2q/100_years_of_hurricane_paths_animated_oc/)

I tried to implement some of the enhancements that users requested in the reddit posting.

That ended up being this kind of thought process:

* Ignored all storm basins except the North American Atlantic Basin as that's where most of the (Western) damage is concentrated.  (However Asian typhoons have been measured to be quite a bit stronger.)
* I took each track and figured out the category of the hurricane at any given time. 
* At each time interval apply some 'damage' to the lat/long that the storm is currently located at.
* The damage formula was: (storm category + 2)^2   This accounted for tropical storms and tropical depressions. The square is because the Saffir-Simpson Hurricane Categories are a roughly logarithmic scale.  Also, according to the Saffir-Simpson scale, wind is the primary source of damage from hurricanes and air pressure increases as the square of velocity.

[Gifs of all mentioned videos here]()

That lead to this gif:
[100 Years of Hurricane Intensities](https://www.youtube.com/watch?v=e71EJGwv8yE)

One can see some interesting trends, like hurricanes rapidly losing strength once they go onshore. But many improvements were possible. 

* To make it less busy I binned the intensities into 0.2* x 0.2* (lat/long) squares.  
* I used a 25 year average, ending on the last year of the interval
* I normalized all of the years heatmaps (gamma) using the values from the highest intensity interval. This helps to see the relative activity by interval.

That resulted in:
[Hurricane 25 Year Moving Average 'Damage' Heat Map](https://www.youtube.com/watch?v=qv1DYSV70ss)

This seemed more useful but storm intensity and detail was lost. Some of the feedback on the original post requested color by strength and category.

* I took each track and determined the 'instantaneous' category using the Wind Speed feature of the data.
* I then used the category data to determine color as well as the size of a scatter plot dot.
* To make the animations smoother I interpolated between time intervals using a cubic algorithm. This lead to smoother color and size graduations.
* I also applied cubic interpolation to the latitude and longitude data so that the tracks were solid rather than dots.
* Because storms have widely varying durations I normalized the durations of all of the storms in one year to the duration of the longest storm. Then I used interpolation to get the correct number of frames for each year. (For making the video)
* Finally I ran the process in succession, saving previous years as white intensity transparent pngs. These were used to show the tracks of the previous 5 years of storms.

This all resulted in:
[Atlantic Hurricane Tracks Animated by Year (fast speed)](https://www.youtube.com/watch?v=lcr8TEFHZqA)  See video link for slower speeds that are more comprehensible.

The 5 years of historical tracks makes things a bit too busy but this particular video takes a long time to render a new one until a later date. The combining color of the tracks was to indicate areas being hit more than once but I think in the future I will simply keep the most 'powerful' color.

Finally just to see what it would look like I made a sort of cinemagraph of 100 years of the Atlantic Hurricane Data.  That was a good programming challenge but didn't turn out as well as I would have liked.  Over the course of a minute all ~1500 storms are activated and put through their tracks in turn.

That looked like the following:
[100 Years of Hurricanes Cinemagraph](https://www.youtube.com/watch?v=brU_P6gFq5w)

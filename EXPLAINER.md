# What this project is, explained from scratch

*No background needed. About ten minutes to read.*

---

## The one-sentence version

Self-driving cars use a laser scanner to see the world in 3D. Rain breaks it.
We built six different programs to clean up the damage, tested them
carefully, and discovered that the standard way of scoring such programs is
misleading.

---

## 1. How a self-driving car "sees"

A car cannot drive on camera images alone. A camera gives you a flat picture,
and a flat picture does not tell you how far away things are. So most
self-driving cars carry a **LiDAR**: a laser scanner that sits on the roof and
spins.

Think of a laser tape measure. You point it at a wall, it fires a pulse of
light, times how long the reflection takes to come back, and tells you the
distance. A LiDAR is the same idea, except it spins around ten times a second
and fires roughly a hundred thousand pulses per rotation, in every direction
at once.

What comes back is not a photo. It is a **point cloud**: a swarm of a hundred
thousand dots floating in 3D space, each one marking a spot where the laser
hit something solid. Cars, walls, kerbs, pedestrians and road surface all show
up as clusters of dots. Software then looks at those clusters and works out
"that shape is a car, and it is eleven metres ahead of us".

If you have seen a picture that looks like a glowing wireframe city made of
dots, that is a point cloud.

---

## 2. What rain does to it

Here is the problem. The laser does not know the difference between a car and
a raindrop. It only knows that *something* reflected its pulse.

Rain damages the scan in two different ways, and they matter for different
reasons.

**Problem one: ghost dots.**
A raindrop drifts into the laser's path. The pulse bounces off the droplet
instead of reaching the car behind it. The scanner faithfully reports a solid
object floating in mid-air two metres away. It is not there. Do this a hundred
thousand times a second in heavy rain and the car is now surrounded by a
snowstorm of dots that do not correspond to anything real.

**Problem two: missing dots.**
Rain also absorbs and scatters light on the way out and on the way back. A
pulse aimed at a distant, dark object comes back too weak to detect at all, so
the scanner reports nothing. The object is simply invisible.

In our simulated rain at a moderate 25 mm per hour, **31% of the real
measurements are destroyed outright.** Nearly a third of the world just stops
existing.

That second problem shapes everything that follows, so it is worth pausing on.
You can delete a ghost dot. You cannot resurrect a measurement that was never
recorded. A cleaning program can tidy up the mess, but it can never get back
to the original clean scan. This turns out to matter a lot for how you judge
whether a cleaner is any good, which is Finding 2 below.

---

## 3. What we built

We took real laser scans recorded by a car driving around a German city on a
clear day. Those are our "clean" reference. Then we ran them through a
physics simulation of rain, based on published measurements of how water
droplets scatter light at the exact wavelength these scanners use. That gives
us matched pairs: the same street, clean and rained on.

Then we built and tested **six different cleaning programs**, of two kinds.

**Three that follow fixed rules.** These have no learning in them. They apply
common sense geometry: "a dot with no neighbours nearby is probably a ghost,
so delete it". Fast, simple, and about as good as their rule is.

**Three that learn.** These are neural networks trained by showing them
thousands of before-and-after examples until they work out for themselves what
rain damage looks like. Slower, but they can pick up patterns nobody
programmed in.

We also defined a **perfect cleaner** as a reference point: an imaginary
program that deletes every single fake dot and keeps every real one, with no
mistakes. It cannot be built, but it tells us what the best possible score
even is.

---

## 4. What we found

### Finding 1: There is a hard ceiling, and one method almost reaches it

Because rain destroys 31% of the measurements permanently, even the perfect
cleaner cannot restore the original scan. It gets a certain distance and stops.
That is the ceiling.

Our best network, a design called DGCNN, gets **98.4% of the way to that
ceiling**. It is essentially as good as it is possible to be. The catch: it
takes 1.8 seconds to clean a single scan, and the scanner produces ten scans
every second. It is roughly twenty times too slow to use on a moving car.

### Finding 2: The standard scoring method is misleading (this is the important one)

Everyone in this field scores a cleaning program by measuring how close the
cleaned scan is to the original clean scan. The measure is called Chamfer
Distance. Lower is better. It sounds obviously correct.

It is not.

One of our programs, an autoencoder, **improved that score** while being
catastrophically bad. When we fed its output to a real car-detection system,
detection collapsed. It found only **6% of the cars** it should have found. The
uncleaned rainy scan did four times better.

Here is the intuition. Imagine judging a restored photograph by measuring the
average colour difference from the original. You could score brilliantly by
smearing the whole image into a smooth blur of roughly the right average
colour. The number improves. The photograph becomes useless. That is exactly
what the autoencoder did to the point cloud: it smoothed everything into a
vague approximate surface, which scores well and destroys the fine structure
the car detector actually relies on.

We found a second, cleaner proof by accident. A bug in our own code made one
program output **232% of the points it was given**, duplicating them. Its
Chamfer score *improved*. You can game the metric simply by producing more
dots.

The lesson: if you are building one of these systems, do not score it on
reconstruction quality alone. Test whether the thing that comes next in the
pipeline, the part that actually finds cars, still works.

### Finding 3: Speed comes from changing how you represent the data

Our accurate network was twenty times too slow, so we built a new one.

The insight is that a spinning LiDAR is not really producing a random cloud of
dots. It has 64 lasers stacked vertically, spinning together. So the data is
naturally a grid: 64 rows, one per laser, and a few thousand columns, one per
angle of rotation. It is shaped like an image.

Once you rearrange it that way, you can use the same kind of fast image
processing that phones use for photos, instead of expensive searches through
3D space asking "which dots are near this dot?"

The result, which we called **RangeRestoreNet**, runs **forty times faster**
(45 milliseconds instead of 1,806) and is the single best method we tested at
correctly identifying which dots are fake. It comfortably keeps up with a real
scanner in real time.

---

## 5. Two things that did not work

Reporting these matters as much as reporting the successes. Both are in the
final report at the same length as the positive results.

**Position correction does nothing.** Both learned models try to nudge slightly
misplaced dots back to where they should be. We measured it. They do not: the
average error moves from 8.04 cm to 7.83 cm, which is nothing. And it turns
out this is not a bug we could fix by training harder. The rain scatters dots
randomly in *all* directions equally. Given one dot, there is no information
about which way it was pushed, so the mathematically optimal guess is "do not
move it at all". The networks correctly learned to do nothing.

**Training without answer keys does not work yet.** Everything above relies on
knowing which dots are fake, because we made them fake. A real car in real
rain has no such answer key. So we tried teaching a network using only rules
of thumb derived from physics ("faint returns with no neighbours are probably
rain"). It learned to copy those rules almost exactly and did not improve on
them at all. The gap between that and the properly trained version is a
precise measurement of how much the answer key was worth.

---

## 6. What we are *not* claiming

- **The rain is simulated, not real.** It is based on published physics rather
  than invented, and the car detector we tested with had never seen our
  simulation. But testing on genuinely rainy recordings is the obvious next
  step and we have not done it.
- **The dataset is small.** 262 scans from two drives on one afternoon in one
  city.
- **We could not test on a real sensor.** The original plan included a
  physical scanner and a Raspberry Pi. That part was not built.

There is also one thing we got wrong and corrected. An earlier version of this
work claimed that simply deleting bad dots was not the best you could do. That
was based on ten test scans without a proper statistical check. When we redid
it with eighty-four measurements, the result **reversed**. We wrote up the
mistake rather than quietly fixing it, because the reason it happened is
genuinely instructive: the measurement was far noisier than we assumed, and we
had not checked.

---

## 7. Where to look next

| If you want | Look at |
|---|---|
| To see it and play with it | **[The website](https://ramyak-sharma.github.io/lidar-denoising-showcase/)**, where you can rotate the 3D scans and drag sliders to clean them yourself |
| The full technical detail | `report/final_report.pdf` |
| The code | `src/lidar_denoise/` |

On the website, the section called "Tune a filter yourself" is running the
real cleaning algorithm live in your browser as you move the sliders. Watching
the dots vanish is the fastest way to understand what all of this is actually
doing.

---

*Capstone Project CPG No. 5, Department of Computer Science and Engineering,
Thapar Institute of Engineering and Technology, Patiala.
Contributors: Jayantika, Krish Agarwal, Ramyak Sharma, Shardool Dhodawat,
Siddhant Singh. Mentor: Dr. Lovi Dhamija.*

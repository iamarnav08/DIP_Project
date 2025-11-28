# DIP_Project
# Latest status
- Use og_equation_solver for anything to be done from now on.
- I have added a script for visualizing optical flow, which we will need for multiple visualizations in our presentation.
- Read the presentation plan I have added on the Notion page.
- READ PAPER
- Understand the visualization method. (VERY IMPORTANT!)
- Need to improve the code a lot, even now.

## The following is old.
trial.py is what I've done till now.
Did not get time to code on my own, just prompted a few things to get results.
So, the current implementation might not be completely faithful to the paper.
If you are done with reading the paper, try to see if you can improve upon it, or just make your version from scratch.


The paper we have taken is SteadyFlow. It operates on the optical flow for every pixel.
The problem with this is:
- computationally expensive. (Took me some time to make something that did not crash my device).
- When you smoothen it out, it kind of warps pixels around. You can see stabilization is being done, but the video is not good.
- By warping, I mean that pixels might be pushed around, making it difficult to get the original video.

https://github.com/how4rd/meshflow

The GitHub link given above is for MeshFlow, the next paper by the same author. It operates on a mesh/grid of points instead of all pixels.
This repo has very good results, I tried it out for myself as well.
Maybe we can use it for inspiration for better results.
I could not find any implementations of SteadyFlow online.

If we can fix it, very well.
If not, we can present our results and proceed with MeshFlow, without explicitly stating what we are doing, but get better results.

To fix it, I'll need to seriously sit with the code, which I can do only from the afternoon of the 27th.
We'll have 3-4 days to finish it.

I have added an example input video I used and the output I got.
You can get other inputs from the meshflow link I've added.

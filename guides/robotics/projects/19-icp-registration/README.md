# ICP Registration

## Key Insight

Two depth scans of the same scene taken from different viewpoints describe the same surfaces, but each in its own coordinate frame — [registration](/shared/glossary/#point-cloud-registration) is the job of finding the rigid [transform](/shared/glossary/#homogeneous-transform) that snaps one onto the other. [Iterative Closest Point (ICP)](/shared/glossary/#iterative-closest-point-icp) solves it with a simple loop: pair each point with the nearest point in the other cloud, compute the rotation and translation that best align those pairs, move, and repeat until the clouds stop sliding. Visualizing the convergence — the two [point clouds](/shared/glossary/#point-cloud) lurching together over a handful of iterations — builds the intuition for ICP's main weakness: it only ever seeks the *nearest* alignment, so without a decent initial guess it happily locks onto the wrong one.

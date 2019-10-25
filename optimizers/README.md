Optimizers for use in this framework should implement the following interface


\_\_init\_\_(distmeasure, transform)
"""Initialize with an object distmeasure implementing the distance interface and a an object transform implementing the transform interface (start point)."""

set\_options(\*\*kwargs)
"""Set any further parameters required e.g. step length, tolerance."""

set\_report\_freq(freq)
"""Set the number of iterations between status reports."""

set\_report\_callback(report\_func, additive)
"""Set the reporting function, allowing the user to add information such as the pyramid level (if additive=True) or completely replace the output."""

get\_transform()
"""Return the final transform (result of the optimization)."""

get\_iteration()
"""Return the index of the most recently completed iteration (probably used by reporting function, also to confirm number of completed iterations at termination."""

get\_value()
"""Return the final distance metric value (result of the optimization)."""

get\_success\_flag()
"""Return any flag (e.g. success flag) relating to a completed optimization."""

get\_termination\_reason()
"""Return a message about the termination of the optimization process."""

get\_value\_history()
"""Return the metric values evaluated during the optimization process."""

optimize()
"""Run the minimization procedure, using parameters already set, and reporting status after specified number of iterations""".

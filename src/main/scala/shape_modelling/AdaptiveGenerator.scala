package shape_modelling

import shape_modelling.MCMC.ShapeParameters

trait AdaptiveGenerator[A] {
  def adapt(iteration : Int, stepOutput : A, logAcceptance : Double) : Unit
}

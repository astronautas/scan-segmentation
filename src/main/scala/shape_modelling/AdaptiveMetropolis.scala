package shape_modelling

import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionRatio}

import scala.math.exp
import scala.util.Random

class AdaptiveMetropolis[A](override val generator: ProposalGenerator[A] with TransitionRatio[A] with AdaptiveGenerator[A],
                            override val evaluator: DistributionEvaluator[A],
                            override val logger: AcceptRejectLogger[A])(implicit override val random: Random) extends MetropolisHastings[A](
  generator: ProposalGenerator[A] with TransitionRatio[A],
  evaluator: DistributionEvaluator[A],
  logger: AcceptRejectLogger[A]) {
  var iteration: Int = -1

  override def next(current: A): A = {
    iteration += 1

    // reference p value
    val currentP = evaluator.logValue(current)
    // proposal
    val proposal = generator.propose(current)
    val proposalP = evaluator.logValue(proposal)
    // transition ratio
    val t = generator.logTransitionRatio(current, proposal)
    // acceptance probability
    val a = proposalP - currentP - t

    // accept or reject
    if (a > 0.0 || random.nextDouble() < exp(a)) {
      logger.accept(current, proposal, generator, evaluator)

      generator.adapt(iteration, current, proposalP)

      proposal
    }
    else {
      logger.reject(current, proposal, generator, evaluator)
      current
    }
  }
}

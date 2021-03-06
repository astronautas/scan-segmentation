package shape_modelling

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{exp, log, sqrt}
import scalismo.geometry.{Point, SquareMatrix, Vector3D, _3D}
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, SymmetricTransition, TransitionProbability}
import scalismo.statisticalmodel.{MultivariateNormalDistribution, NDimensionalNormalDistribution, StatisticalMeshModel}

object MCMC {

  case class ShapeParameters(rotationParameters: DenseVector[Float], translationParameters: DenseVector[Float], modelCoefficients: DenseVector[Float])

  case class RotationUpdateProposal(stdev: Float) extends
    ProposalGenerator[ShapeParameters] with TransitionProbability[ShapeParameters] with SymmetricTransition[ShapeParameters] {

    val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(3),
      DenseMatrix.eye[Float](3) * stdev)

    override def propose(theta: ShapeParameters): ShapeParameters = {
      val perturbation = perturbationDistr.sample()

      val thetaPrime = ShapeParameters(theta.rotationParameters + perturbation, theta.translationParameters, theta.modelCoefficients)
      thetaPrime
    }

    override def logTransitionProbability(from: ShapeParameters, to: ShapeParameters): Double = {
      val residual = to.rotationParameters - from.rotationParameters
      perturbationDistr.logpdf(residual)
    }
  }

  case class RotationUpdateProposalX(stdev: Float) extends
    ProposalGenerator[ShapeParameters] with TransitionProbability[ShapeParameters] with SymmetricTransition[ShapeParameters] {

    val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(3),
      DenseMatrix.eye[Float](3) * stdev)

    override def propose(theta: ShapeParameters): ShapeParameters = {
      val perturbation = perturbationDistr.sample()

      perturbation(1) = 0
      perturbation(2) = 0

      val thetaPrime = ShapeParameters(theta.rotationParameters + perturbation, theta.translationParameters, theta.modelCoefficients)
      thetaPrime
    }

    override def logTransitionProbability(from: ShapeParameters, to: ShapeParameters): Double = {
      val residual = to.rotationParameters - from.rotationParameters
      perturbationDistr.logpdf(residual)
    }
  }

  case class TranslationUpdateProposal(stdev: Float) extends
    ProposalGenerator[ShapeParameters] with TransitionProbability[ShapeParameters] with SymmetricTransition[ShapeParameters] {

    val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(3),
      DenseMatrix.eye[Float](3) * stdev)

    def propose(theta: ShapeParameters): ShapeParameters = {
      val perturbation = perturbationDistr.sample()
      val thetaPrime = ShapeParameters(theta.rotationParameters, theta.translationParameters + perturbation, theta.modelCoefficients)
      thetaPrime
    }

    override def logTransitionProbability(from: ShapeParameters, to: ShapeParameters): Double = {
      val residual = to.translationParameters - from.translationParameters
      perturbationDistr.logpdf(residual)
    }
  }

  case class ShapeUpdateProposal(paramVectorSize: Int, stdev: Float) extends
    ProposalGenerator[ShapeParameters] with TransitionProbability[ShapeParameters] with SymmetricTransition[ShapeParameters] {

    val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(paramVectorSize),
      DenseMatrix.eye[Float](paramVectorSize) * stdev)

    override def propose(theta: ShapeParameters): ShapeParameters = {
      val perturbation = perturbationDistr.sample()

      val thetaPrime = ShapeParameters(theta.rotationParameters, theta.translationParameters, theta.modelCoefficients + perturbation)
      thetaPrime
    }

    override def logTransitionProbability(from: ShapeParameters, to: ShapeParameters) = {
      val residual = to.modelCoefficients - from.modelCoefficients
      perturbationDistr.logpdf(residual)
    }
  }

  // ADAPTIVE SHAPE UPDATE PROPOSAL
  case class ShapeUpdateProposalAdapting(paramVectorSize: Int, stdev: Float, adaptScale: Boolean = false, sampleDiscard: Int,
                                         sampleLag: Int, accStar: Double) extends
    ProposalGenerator[ShapeParameters] with TransitionProbability[ShapeParameters] with SymmetricTransition[ShapeParameters] with AdaptiveGenerator[ShapeParameters] {

    var perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(paramVectorSize),
      DenseMatrix.eye[Float](paramVectorSize) * stdev)

    var globalScale: Double = Math.pow(2.38, 2) / paramVectorSize
    var covEst: DenseMatrix[Float] = DenseMatrix.eye[Float](paramVectorSize) * stdev
    var meanEst: DenseVector[Float] = DenseVector.ones(paramVectorSize)
    var learnScale: Float = 0f

    override def propose(theta: ShapeParameters): ShapeParameters = {
      perturbationDistr = new MultivariateNormalDistribution(theta.modelCoefficients, cov = globalScale.toFloat * covEst)

      val perturbation = perturbationDistr.sample()

      val thetaPrime = ShapeParameters(theta.rotationParameters, theta.translationParameters, perturbationDistr.sample)
      thetaPrime
    }

    override def logTransitionProbability(from: ShapeParameters, to: ShapeParameters): Double = {
      val residual = to.modelCoefficients - from.modelCoefficients
      perturbationDistr.logpdf(residual)
    }

    // ADAPTING
    def adapt(iteration: Int, stepOutput: ShapeParameters, logAcceptance: Double): Unit = {

      if (iteration > sampleDiscard) {
        learnScale = (1.0 / sqrt(iteration - sampleDiscard + 1.0)).toFloat

        if (adaptScale) {
          scaleAdapt(learnScale, stepOutput, logAcceptance)
        }

        if (iteration % sampleLag == 0) {
          meanAndCovAdapt(learnScale, stepOutput)
        }
      }
    }

    def meanAndCovAdapt(learnScale: Float, stepOutput: ShapeParameters): Unit = {
      var diff: DenseVector[Float] = DenseVector.ones(paramVectorSize)

      // implement this
      var current = stepOutput.modelCoefficients

      diff = current - meanEst

      meanEst += learnScale * diff
      covEst += learnScale * (diff.t * diff - covEst)
    }

    def scaleAdapt(learn_scale: Double, step_output: ShapeParameters, logAcceptance: Double): Unit = {
      // implement this
      //        self.globalscale = exp(log(self.globalscale) + learn_scale * (exp(step_output.log_ratio) - self.accstar))
      globalScale = exp(log(globalScale) + learn_scale + (exp(logAcceptance) - accStar))
    }
  }

  case class ShapeUpdateProposalFirstComponents(paramVectorSize: Int, stdev: Float, componentIndex: Int) extends
    ProposalGenerator[ShapeParameters] with TransitionProbability[ShapeParameters] with SymmetricTransition[ShapeParameters] {

    val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(paramVectorSize),
      DenseMatrix.eye[Float](paramVectorSize) * stdev)

    override def propose(theta: ShapeParameters): ShapeParameters = {
      // Generate distribution
      val perturbation = perturbationDistr.sample()

      0 until perturbation.length foreach (index => if (componentIndex != index) perturbation(index) = 0)

      val thetaPrime = ShapeParameters(theta.rotationParameters, theta.translationParameters, theta.modelCoefficients + perturbationDistr.sample)
      thetaPrime
    }

    override def logTransitionProbability(from: ShapeParameters, to: ShapeParameters) = {
      val residual = to.modelCoefficients - from.modelCoefficients
      perturbationDistr.logpdf(residual)
    }
  }

  case class ProximityEvaluator(model: StatisticalMeshModel, targetLandmarks: Seq[Point[_3D]],
                                sdev: Double = 1.0) extends DistributionEvaluator[ShapeParameters] {

    val uncertainty = NDimensionalNormalDistribution(Vector3D(0f, 0f, 0f), SquareMatrix.eye[_3D] * (sdev * sdev))

    override def logValue(theta: ShapeParameters): Double = {

      val currModelInstance = model.instance(theta.modelCoefficients)

      val likelihoods = targetLandmarks.map { targetLandmark =>

        val closestPointCurrentFit = currModelInstance.findClosestPoint(targetLandmark).point
        val observedDeformation = targetLandmark - closestPointCurrentFit
        uncertainty.logpdf(observedDeformation)
      }

      val loglikelihood = likelihoods.sum
      loglikelihood
    }
  }

  // Check how likely the prior is (concrete instance from params), with a concern to the model
  case class ShapePriorEvaluator(model: StatisticalMeshModel) extends DistributionEvaluator[ShapeParameters] {
    override def logValue(theta: ShapeParameters): Double = {
      //println("ShapePriorEvaluator: returning "+model.gp.logpdf(theta.modelCoefficients))
      model.gp.logpdf(theta.modelCoefficients)
    }
  }

}

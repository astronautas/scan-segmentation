package shape_modelling

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.geometry.{Point, SquareMatrix, Vector3D, _3D}
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, SymmetricTransition, TransitionProbability}
import scalismo.statisticalmodel.asm.{ActiveShapeModel, PreprocessedImage}
import scalismo.statisticalmodel.{MultivariateNormalDistribution, NDimensionalNormalDistribution, StatisticalMeshModel}

object MCMC {

  case class ShapeParameters(rotationParameters: DenseVector[Float], translationParameters: DenseVector[Float], modelCoefficients: DenseVector[Float])

  case class RotationUpdateProposal(stdev: Float) extends
    ProposalGenerator[ShapeParameters] with TransitionProbability[ShapeParameters] with SymmetricTransition[ShapeParameters] {

    val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(3),
      DenseMatrix.eye[Float](3) * stdev)

    def propose(theta: ShapeParameters): ShapeParameters = {
      ShapeParameters(theta.rotationParameters + perturbationDistr.sample, theta.translationParameters, theta.modelCoefficients)
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
      ShapeParameters(theta.rotationParameters + perturbationDistr.sample, theta.translationParameters, theta.modelCoefficients)
    }

    override def logTransitionProbability(from: ShapeParameters, to: ShapeParameters): Double = {
      val residual = to.translationParameters - from.translationParameters
      perturbationDistr.logpdf(residual)
    }
  }

  case class ShapeUpdateProposal(paramVectorSize: Int, stdev: Float) extends
    ProposalGenerator[ShapeParameters] with TransitionProbability[ShapeParameters] with SymmetricTransition[ShapeParameters] {

    var paramVSize = paramVectorSize
    var stdevThis = stdev

    var perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(paramVectorSize),
      DenseMatrix.eye[Float](paramVectorSize) * stdev)

    var annealingOps = 40f

    var annealingePerturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(paramVSize),
      DenseMatrix.eye[Float](paramVSize) * stdevThis * annealingOps)

    // TODO - here's sort of step that initially makes huge "jumps", and then cooldowns to normal jumps
    // should help for fastening convergence
    override def propose(theta: ShapeParameters): ShapeParameters = {

      if (annealingOps != 0) {
        annealingePerturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(paramVSize),
          DenseMatrix.eye[Float](paramVSize) * stdevThis * annealingOps)

        annealingOps -= 1

        val perturbation = annealingePerturbationDistr.sample()
        val thetaPrime = ShapeParameters(theta.rotationParameters, theta.translationParameters, theta.modelCoefficients + perturbation)

        thetaPrime
      } else {
        val perturbation = perturbationDistr.sample()
        val thetaPrime = ShapeParameters(theta.rotationParameters, theta.translationParameters, theta.modelCoefficients + perturbation)
        thetaPrime
      }
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

  case class IntensityBasedLikeliHoodEvaluator(asm: ActiveShapeModel, preprocessedImage: PreprocessedImage,
                                               sdev: Double = 1.0) extends DistributionEvaluator[ShapeParameters] {

    val uncertainty = NDimensionalNormalDistribution(Vector3D(0f, 0f, 0f), SquareMatrix.eye[_3D] * (sdev * sdev))

    override def logValue(theta: ShapeParameters): Double = {
      val t1 = System.nanoTime()

      val value = LikelihoodChecker.likelihoodThatMeshFitsImage(asm, asm.statisticalModel.instance(theta.modelCoefficients), preprocessedImage)

      val time = (System.nanoTime() - t1) / 1000000000.0
      System.out.println("[TIME] LC time: " + time)

      value
    }
  }

  // Check how likely the prior is (concrete instance from params), with a concern to the model
  case class ShapePriorEvaluator(model: StatisticalMeshModel) extends DistributionEvaluator[ShapeParameters] {
    override def logValue(theta: ShapeParameters): Double = {
      model.gp.logpdf(theta.modelCoefficients)
    }
  }

}

package shape_modelling

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import breeze.linalg.DenseVector
import scalismo.geometry._3D
import scalismo.image.DiscreteScalarImage
import scalismo.io.{ActiveShapeModelIO, ImageIO}
import scalismo.numerics.UniformSampler
import scalismo.registration.RigidTransformationSpace
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator}
import scalismo.statisticalmodel.asm.{ActiveShapeModel, PreprocessedImage}
import scalismo.ui.ShapeModelView
import scalismo.ui.api.SimpleAPI.ScalismoUI
import shape_modelling.MCMC._

import scala.util.Random

object SegmentationNonUI {
  private[this] var ui: ScalismoUI = _
  var best_prob_ever = 0.0

  def main(args: Array[String]) {
		// required to initialize native libraries (VTK, HDF5 ..)
		scalismo.initialize()

		// Your application code goes below here. Below is a dummy application that reads a mesh and displays it

		// create a visualization window
		//ui = ScalismoUI()

		val asm = ActiveShapeModelIO.readActiveShapeModel(new File("handedData/femur-asm.h5")).get
		val image = ImageIO.read3DScalarImage[Short](new File("handedData/targets/37.nii")).get.map(_.toFloat)

		//ui.show(asm.statisticalModel, "model")
		//ui.show(image, "image")

		// PREPROCESSING
		System.out.println("Preprocessing...")
		val prepImg: PreprocessedImage = asm.preprocessor(image)

		// FITTING
		//runPoseFitting(asm, image)
		var coeffs = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm.statisticalModel.coefficients(asm.statisticalModel.mean))

		0 until coeffs.modelCoefficients.length / 20 foreach(i => {
			coeffs = runShapeFittingForComponents(asm, prepImg, coeffs, i)
		})

		coeffs = runShapeFitting(asm, prepImg, coeffs)
  }

  def runPoseFitting(asm: ActiveShapeModel, image: DiscreteScalarImage[_3D, Float]): Unit = {
	val samples = UniformSampler(image.domain.boundingBox, 1000).sample.map(i => i._1)

	System.out.println("Running pose fitting...")

	val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), ProximityEvaluator(asm.statisticalModel, samples))

	val poseGenerator = MixtureProposal.fromProposalsWithTransition((0.2, RotationUpdateProposal(asm.statisticalModel.rank, 0.1f)), (0.2, TranslationUpdateProposal(asm.statisticalModel.rank, 0.1f)))(rnd = new Random())
	val chain = MetropolisHastings(poseGenerator, posteriorEvaluator)(new Random())

	val initialParameters = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), DenseVector.zeros[Float](asm.statisticalModel.rank))
	val mhIt = chain.iterator(initialParameters)

	val rigidTransSpace = RigidTransformationSpace[_3D]()
	var bestCoefs: ShapeParameters = null
	var bestProb: Double = 0
	val samplingIterator = for (theta <- mhIt) yield {

	  val prob = posteriorEvaluator.logValue(theta)
	  System.out.println(prob)

	  if (prob > bestProb) {
		System.out.println("Found!!")

		ui.setCoefficientsOf("model", theta.modelCoefficients)
		val rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(theta.translationParameters, theta.rotationParameters))


		// internal scalismo ui code for efficiency reasons, to update the rigid position of the model (no need to understand the details of this)
		val smInstance = ui.frame.scene.find[ShapeModelView](si => si.name == "model").head.instances.head
		smInstance.rigidTransformation = Some(rigidtrans)

		bestCoefs = theta
		bestProb = prob
	  }

	}

	samplingIterator.take(40).toIndexedSeq
  }

  def runPositionFitting(asm: ActiveShapeModel, image: DiscreteScalarImage[_3D, Float]): Unit = {
	val samples = UniformSampler(image.domain.boundingBox, 1000).sample.map(i => i._1)

	System.out.println("Running position fitting...")

	val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), ProximityEvaluator(asm.statisticalModel, samples))

	val poseGenerator = MixtureProposal.fromProposalsWithTransition((0.2, RotationUpdateProposal(asm.statisticalModel.rank, 0.1f)), (0.2, TranslationUpdateProposal(asm.statisticalModel.rank, 0.1f)))(rnd = new Random())
	val chain = MetropolisHastings(poseGenerator, posteriorEvaluator)(new Random())

	val initialParameters = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), DenseVector.zeros[Float](asm.statisticalModel.rank))
	val mhIt = chain.iterator(initialParameters)

	val rigidTransSpace = RigidTransformationSpace[_3D]()
	var bestCoefs: ShapeParameters = null
	var bestProb: Double = 0
	val samplingIterator = for (theta <- mhIt) yield {

	  val prob = posteriorEvaluator.logValue(theta)
	  System.out.println(prob)

	  if (prob > bestProb) {
		System.out.println("Found!!")

		ui.setCoefficientsOf("model", theta.modelCoefficients)
		val rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(theta.translationParameters, theta.rotationParameters))


		// internal scalismo ui code for efficiency reasons, to update the rigid position of the model (no need to understand the details of this)
		val smInstance = ui.frame.scene.find[ShapeModelView](si => si.name == "model").head.instances.head
		smInstance.rigidTransformation = Some(rigidtrans)

		bestCoefs = theta
		bestProb = prob
	  }

	}

	samplingIterator.take(40).toIndexedSeq
  }

  def runPositionFittingVariable(asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, variance_rot: Float, variance_trans: Float, takeSize: Int): Unit = {
	//val samples = UniformSampler(image.domain.boundingBox, 1000).sample.map(i => i._1)

    val logger = new AcceptRejectLogger[ShapeParameters] {
      private var accepted = 0f
      private var all = 0f

      override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
        accepted += 1
        all += 1

        val ratio = accepted / all
        best_prob_ever = evaluator.logValue(sample)
        println(s"Accepted proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
      }

      override def reject(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
        all += 1
        val ratio = accepted / all
        println(s"Rejected proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
      }
    }

	System.out.println("Running position fitting...")

	val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), IntensityBasedLikeliHoodEvaluator(asm, prepImg))
	val positionGenerator = MixtureProposal.fromProposalsWithTransition((0.5, RotationUpdateProposal(asm.statisticalModel.rank, variance_rot)), (0.5, TranslationUpdateProposal(asm.statisticalModel.rank, variance_trans)))(rnd = new Random())

	val chain = MetropolisHastings(positionGenerator, posteriorEvaluator, logger)(new Random())
	val mhIt = chain.iterator(initialParameters)

	val rigidTransSpace = RigidTransformationSpace[_3D]()
	var bestCoeffs: ShapeParameters = null
	var bestProb: Double = 0
	val samplingIterator = for (theta <- mhIt) yield {
	  val prob = posteriorEvaluator.logValue(theta)
	  if (prob > bestProb) {
		ui.setCoefficientsOf("model_cyclic_repositioning", theta.modelCoefficients)
		val rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(theta.translationParameters, theta.rotationParameters))

		// internal scalismo ui code for efficiency reasons, to update the rigid position of the model (no need to understand the details of this)
		val smInstance = ui.frame.scene.find[ShapeModelView](si => si.name == "model_cyclic_repositioning").head.instances.head
		smInstance.rigidTransformation = Some(rigidtrans)

		bestCoeffs = theta
		bestProb = prob
	  }
	}
  }

  def runShapeFittingForComponents(asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, tillComponentIndex: Int): ShapeParameters = {
	val logger = new AcceptRejectLogger[ShapeParameters] {
	  private var accepted = 0f
	  private var all = 0f

	  override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
		accepted += 1
		all += 1

		val ratio = accepted / all
		println(s"Accepted proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
	  }

	  override def reject(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
		all += 1
		val ratio = accepted / all
		println(s"Rejected proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
	  }
	}


	println("Running runShapeFittingForFirstComponents till..." + tillComponentIndex)

	val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), IntensityBasedLikeliHoodEvaluator(asm, prepImg))

	// Deviations should match deviations of model
	val poseGenerator = MixtureProposal.fromProposalsWithTransition((0.8, ShapeUpdateProposalFirstComponents(asm.statisticalModel.rank, 0.09f, tillComponentIndex)), (0.2, ShapeUpdateProposalFirstComponents(asm.statisticalModel.rank, 0.2f, tillComponentIndex)))(rnd = new Random())

	val chain = MetropolisHastings(poseGenerator, posteriorEvaluator, logger)(new Random())

	val mhIt = chain.iterator(initialParameters)

	val samplingIterator = for (theta <- mhIt) yield {
	  theta
	}

	// TODO - what is burn in factor, how to get rid of it (thats why drop is here)
	// http://background.uchicago.edu/~whu/Courses/Ast321_11/Projects/mcmc_helsby.pdf
	val take = 200
	val samples = samplingIterator.drop(take / 12).take(take)
	val bestSample = samples.maxBy(posteriorEvaluator.logValue)

	bestSample
  }

  def runShapeFitting(asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters): ShapeParameters = {
	val logger = new AcceptRejectLogger[ShapeParameters] {
	  private var accepted = 0f
	  private var all = 0f

	  override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
		accepted += 1
		all += 1

		val ratio = accepted / all
		println(s"Accepted proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
	  }

	  override def reject(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
		all += 1
		val ratio = accepted / all
		println(s"Rejected proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
	  }
	}

	System.out.println("Running runShapeFitting...")

	val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), IntensityBasedLikeliHoodEvaluator(asm, prepImg))

	// Deviations should match deviations of model
	val poseGenerator = MixtureProposal.fromProposalsWithTransition((0.7, ShapeUpdateProposal(asm.statisticalModel.rank, 0.1f)))(rnd = new Random())

	val chain =  MetropolisHastings[ShapeParameters](poseGenerator, posteriorEvaluator, logger)(new Random())

	val mhIt = chain.iterator(initialParameters)

	val samplingIterator = for (theta <- mhIt) yield {
	  theta
	}

	val samples = samplingIterator.take(2000).toIndexedSeq

	samples.maxBy(posteriorEvaluator.logValue)
  }

  def runShapeFittingVariable(asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, variance: Float, takeSize: Int): ShapeParameters = {
	val logger = new AcceptRejectLogger[ShapeParameters] {
	  private var accepted = 0f
	  private var all = 0f

	  override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
		accepted += 1
		all += 1

		val ratio = accepted / all
        best_prob_ever = evaluator.logValue(sample)
		println(s"Accepted proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
	  }

	  override def reject(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
		all += 1
		val ratio = accepted / all
		println(s"Rejected proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
	  }
	}

	System.out.println("Running runShapeFitting...")

	val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), IntensityBasedLikeliHoodEvaluator(asm, prepImg))
	// Deviations should match deviations of model
	val poseGenerator = MixtureProposal.fromProposalsWithTransition((1.0, ShapeUpdateProposal(asm.statisticalModel.rank, variance)))(rnd = new Random())

	val chain =  MetropolisHastings[ShapeParameters](poseGenerator, posteriorEvaluator, logger)(new Random())
	val mhIt = chain.iterator(initialParameters)

	val samplingIterator = for (theta <- mhIt) yield {
	  theta
	}

	val samples = samplingIterator.drop(takeSize/10).take(takeSize).toIndexedSeq
	samples.maxBy(posteriorEvaluator.logValue)
  }

  def runShapeFittingWithAdapting(asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters): ShapeParameters = {
	val logger = new AcceptRejectLogger[ShapeParameters] {
	  private var accepted = 0f
	  private var all = 0f

	  override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
		accepted += 1
		all += 1

		val ratio = accepted / all
		println(s"Accepted proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
	  }

	  override def reject(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
		all += 1
		val ratio = accepted / all
		println(s"Rejected proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
	  }
	}

	System.out.println("Running runShapeFitting...")

	val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), IntensityBasedLikeliHoodEvaluator(asm, prepImg))

	// Deviations should match deviations of model
	val proposal = ShapeUpdateProposalAdapting(paramVectorSize = asm.statisticalModel.rank, stdev = 0.2f, sampleDiscard = 20, sampleLag = 2, accStar = 0.234)
	//val poseGenerator = MixtureProposal.fromProposalsWithTransition((0.7, proposal))(rnd = new Random())

	val chain =  new AdaptiveMetropolis[ShapeParameters](proposal, posteriorEvaluator, logger)(new Random())

	val mhIt = chain.iterator(initialParameters)

	val samplingIterator = for (theta <- mhIt) yield {
	  theta
	}

	val samples = samplingIterator.take(2000).toIndexedSeq

	samples.maxBy(posteriorEvaluator.logValue)
  }
}

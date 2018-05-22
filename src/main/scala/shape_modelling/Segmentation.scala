package shape_modelling

import java.io.{File, PrintWriter}
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

object Segmentation {
  private[this] var ui: ScalismoUI = _
  var best_prob_ever = 0.0

  def main(args: Array[String]) {


	// required to initialize native libraries (VTK, HDF5 ..)
	scalismo.initialize()

	// Your application code goes below here. Below is a dummy application that reads a mesh and displays it

	// create a visualization window
	ui = ScalismoUI()

	val targetname = "37"
    val asm_orig_aligned = ActiveShapeModelIO.readActiveShapeModel(new File("handedData/femur-asm.h5")).get
	val asm_for_cyclic_exclusive = asm_orig_aligned
    val asm_for_cyclic_repositioning = asm_orig_aligned
	val image = ImageIO.read3DScalarImage[Short](new File(s"handedData/targets/$targetname.nii")).get.map(_.toFloat)

	ui.show(asm_orig_aligned.statisticalModel, "model_old")
	ui.show(image, "image")

	// PREPROCESSING
	System.out.println("Preprocessing...")
	val prepImg: PreprocessedImage = asm_for_cyclic_exclusive.preprocessor(image)

	//Ignore this, just manually imported the vector
	//val coefficients = DenseVector(-1.3055264f, -0.28966737f, -0.07237204f, -0.00410711f, -0.8697963f, -1.2341552f, -1.7041032f, 0.98775864f, 0.49453974f, -0.6134522f, -0.04279228f, 1.0539454f, -2.2465842f, 0.09945868f, 0.1596679f, 0.377265f, 0.5554652f, 0.2209542f, -0.30257356f, 0.5070663f, 0.05105215f, 1.3191476f, 0.5862405f, 1.8405728f, 1.3931898f, 1.0135989f)

	// FITTING
	//runPoseFitting(asm, image)

	/*0 until coeffs.modelCoefficients.length / 20 foreach(i => {
	  coeffs = runShapeFittingForComponents(asm, prepImg, coeffs, i)
	})*/


	// Cyclic exclusive shape fitting: Just cycle over a set of variances several times.
	println("-------------Testing Cyclic exclusive Shape fitting-------------------------")
    var coeffs_cyclic_exclusive = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm_for_cyclic_exclusive.statisticalModel.coefficients(asm_for_cyclic_exclusive.statisticalModel.mean))
	for (outer <- 1 to 10) {
	  println("Running iteration " + outer)
	  for (i <- outer to 10) {
		println("Running iteration " + outer + " with variance " + (0.5f/(i*i)) + " and takeSize " + 500)
		coeffs_cyclic_exclusive = runShapeFittingVariable(asm_for_cyclic_exclusive, prepImg, coeffs_cyclic_exclusive, 0.5f/(i*i), 500)
		println("-----------------------------DONE--------------------------------------")
	  }
	}
	var result = coeffs_cyclic_exclusive.rotationParameters.toString() + "\n" + coeffs_cyclic_exclusive.translationParameters.toString() + "\n" + coeffs_cyclic_exclusive.modelCoefficients.toString()
	Files.write(Paths.get(s"exclusive_coeffs_for_$targetname.txt"), result.getBytes(StandardCharsets.UTF_8))
	ui.show(asm_for_cyclic_exclusive.statisticalModel, "model_cyclic_exclusive")
	ui.setCoefficientsOf("model_cyclic_exclusive", coeffs_cyclic_exclusive.modelCoefficients)
	val best_prob_exclusive = best_prob_ever
	best_prob_ever = 0.0
	println("-----------------Done with Cyclic exclusive Shape fitting--------------------------")


	// Single component fitting. Iterates over each Principal Component, fitting mostly that one component.



	// Cyclic shape fitting with Repositioning: Do normal Cyclic shape fitting, but add position fitting after each outer iteration.
	println("-------------Testing Cyclic Shape fitting with Repositioning-------------------------")
    var coeffs_cyclic_repositioning = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm_for_cyclic_exclusive.statisticalModel.coefficients(asm_for_cyclic_exclusive.statisticalModel.mean))
	for (outer <- 1 to 10) {
	  println("Running iteration " + outer)
	  for (i <- outer to 10) {
		println("Running iteration " + outer + " with variance " + (0.5f/(i*i)) + " and takeSize " + 500)
		coeffs_cyclic_repositioning = runShapeFittingVariable(asm_for_cyclic_repositioning, prepImg, coeffs_cyclic_repositioning, 0.5f/(i*i), 500)
		println("-----------------------------DONE--------------------------------------")
	  }
	  runPositionFittingVariable(asm_for_cyclic_repositioning, prepImg, coeffs_cyclic_repositioning, 0.05f, 0.05f, 20)
	}

    result = coeffs_cyclic_repositioning.rotationParameters.toString() + "\n" + coeffs_cyclic_repositioning.translationParameters.toString() + "\n" + coeffs_cyclic_repositioning.modelCoefficients.toString()
    Files.write(Paths.get(s"repositioning_coeffs_for_$targetname.txt"), result.getBytes(StandardCharsets.UTF_8))
    ui.show(asm_for_cyclic_repositioning.statisticalModel, "model_cyclic_repositioning")
    ui.setCoefficientsOf("model_cyclic_repositioning", coeffs_cyclic_repositioning.modelCoefficients)
    val best_prob_repositioning = best_prob_ever
    best_prob_ever = 0.0
    println("-----------------Done with Cyclic Shape fitting with Repositioning--------------------------")

    println("Best prob for exclusive method: " + best_prob_exclusive)
    println("Best prob for repositioning method: " + best_prob_repositioning)

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

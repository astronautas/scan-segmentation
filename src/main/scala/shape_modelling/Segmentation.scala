package shape_modelling

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.geometry
import scalismo.geometry.{Point3D, Vector3D, _3D}
import scalismo.image.DiscreteScalarImage
import scalismo.io.{ActiveShapeModelIO, ImageIO, LandmarkIO, MeshIO}
import scalismo.numerics.UniformSampler
import scalismo.registration.{RigidTransformation, RigidTransformationSpace}
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
	var best_prob_ever = 0.0
	var center_of_mass : DenseVector[Float] = DenseVector.zeros[Float](3)
	var landmarks_on_asm: Seq[PointId] = null
	var landmarks_on_target: Seq[Point3D] = null

	var asm_orig_unaligned: ActiveShapeModel = _

	private[this] var ui: ScalismoUI = _

	var load_fitted_asm: Boolean = false

	def main(args: Array[String]) {
		if (args.length == 0) {
		  println("Provide the root path for handed data (i.e. handedData)")
		  System.exit(0)
		}

		// required to initialize native libraries (VTK, HDF5 ..)
		scalismo.initialize()

		// Your application code goes below here. Below is a dummy application that reads a mesh and displays it

		// create a visualization window
		ui = ScalismoUI()


		val handedDataRootPath = args(0)

		val targetname = "37"
		val asm_orig_aligned = ActiveShapeModelIO.readActiveShapeModel(new File(s"$handedDataRootPath/femur-asm_aligned.h5")).get
		asm_orig_unaligned = ActiveShapeModelIO.readActiveShapeModel(new File(s"$handedDataRootPath/femur-asm.h5")).get
		val asm_for_cyclic_exclusive = asm_orig_aligned
		val asm_for_cyclic_repositioning = asm_orig_aligned
		val image = ImageIO.read3DScalarImage[Short](new File(s"$handedDataRootPath/targets/$targetname.nii")).get.map(_.toFloat)

		calculate_CoM(asm_orig_unaligned)


		ui.show(asm_orig_aligned.statisticalModel, "original_model")
		ui.show(image, "image")

		// Landmark stuff, please ignore
		/*val asm_lms = LandmarkIO.readLandmarksJson[_3D](new File (s"$handedDataRootPath/femur-landmarks.json")).get
		val target_lms = LandmarkIO.readLandmarksJson[_3D](new File (s"$handedDataRootPath/targets/$targetname.json")).get

		ui.addLandmarksTo(asm_lms, "original_model")
		ui.addLandmarksTo(target_lms, "image")

		landmarks_on_asm = ui.getLandmarksOf("original_model")
		landmarks_on_target = ui.getLandmarksOf("image")*/


		// PREPROCESSING
		System.out.println("Preprocessing...")
		val prepImg: PreprocessedImage = asm_for_cyclic_exclusive.preprocessor(image)

		// FITTING
		//runPoseFitting(asm, image)

		/*0 until coeffs.modelCoefficients.length / 20 foreach(i => {
		  coeffs = runShapeFittingForComponents(asm, prepImg, coeffs, i)
		})*/


		var pose_fitted_asm: ActiveShapeModel = null
		var coeffs_test: ShapeParameters = null
		var take_size = 500
		var variance_rot = 0.0005f
		var variance_trans = 0.001f
		var current_translation_coeffs: DenseVector[Float] = null
		var current_CoM: Point3D = null
		var rigidTransSpace: RigidTransformationSpace[_3D] = null
		var rigidtrans: RigidTransformation[_3D] = null

		if (!load_fitted_asm) {
			println("-------------Doing Pose fitting-------------------------")
			var coeffs_test = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm_orig_unaligned.statisticalModel.coefficients(asm_orig_unaligned.statisticalModel.mean))
			ui.show(asm_orig_unaligned.statisticalModel, "model_updating")

			println("Running iteration pose fitting with variances rot/trans " + variance_rot + "/" + variance_trans + " and take_size " + take_size)
			coeffs_test = runPositionFittingVariable(asm_orig_unaligned, prepImg, coeffs_test, variance_rot, variance_trans, take_size, 0)

			// Saving of pose-fitted asm
			println("-----------------------------Saving pose fitted ASM--------------------------------------")
			// Creating the transformation according to the pose coefficients and rigid transforming the ASM
			val current_translation_coeffs = center_of_mass + coeffs_test.translationParameters
			current_CoM = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))
			rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
			rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(coeffs_test.translationParameters, coeffs_test.rotationParameters))
			pose_fitted_asm = asm_orig_unaligned.transform(rigidtrans)

			ActiveShapeModelIO.writeActiveShapeModel(pose_fitted_asm, new File(s"$handedDataRootPath/femur-asm_pose_fitted.h5"))
			println("-----------------------------ASM saved--------------------------------------")
			println("-----------------------------Pose Fitting done--------------------------------------")
		}

		println("-----------------------------Doing Shape fitting--------------------------------------")
		// We work with the pose fitted ASM now, so reset the coefficients
		if (load_fitted_asm) {
			pose_fitted_asm = ActiveShapeModelIO.readActiveShapeModel(new File(s"$handedDataRootPath/femur-asm_pose_fitted.h5")).get
		}
		ui.remove("model_updating")
		ui.show(pose_fitted_asm, "model_updating")

		coeffs_test = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), pose_fitted_asm.statisticalModel.coefficients(pose_fitted_asm.statisticalModel.mean))
		take_size = 100
		variance_rot /= 10
		variance_trans /= 10
		// Not doing outer iteration right now. I suspect that it's actually counterproductive (time would be better spent on fine-detail fitting).
		for (outer <- 1 to 1) {
			println("-------------------------Running outer iteration " + outer + "----------------------------------")
			for (i <- outer to 10) {
				val variance = 0.1f / (i * i)
				println("Running iteration " + outer + " with variance " + variance + " and takeSize " + take_size)
				coeffs_test = runShapeFittingVariable(pose_fitted_asm, prepImg, coeffs_test, variance, take_size)
				println("-----------------------------Doing intermediate pose fitting--------------------------------------")
				coeffs_test = runPositionFittingVariable(pose_fitted_asm, prepImg, coeffs_test, variance_rot, variance_trans, take_size, 0)
				println("-----------------------------Iteration done--------------------------------------")
			}
		}
		println("-----------------Shape Fitting done--------------------------")

		// Saving coefficient vectors and resulting mesh.
		val result = coeffs_test.rotationParameters.toString() + "\n" + coeffs_test.translationParameters.toString() + "\n" + coeffs_test.modelCoefficients.toString()
		Files.write(Paths.get(s"test_coeffs_for_$targetname.txt"), result.getBytes(StandardCharsets.UTF_8))

		var final_mesh = pose_fitted_asm.statisticalModel.instance(coeffs_test.modelCoefficients)

		// This time we need to transform using the CoM of the NEW asm, i.e. of the pose_fitted_asm
		calculate_CoM(pose_fitted_asm)
		// Figure out transform based on coeffs from the shape fitting phase. They should be all 0 right now, but if we ever introduce pose fitting stuff
		// into the shape fitting phase, this here will come in handy.
		current_translation_coeffs = center_of_mass + coeffs_test.translationParameters
		current_CoM = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))
		rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
		rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(coeffs_test.translationParameters, coeffs_test.rotationParameters))

		final_mesh = final_mesh.transform(rigidtrans)
		MeshIO.writeMesh(final_mesh, new File(s"$handedDataRootPath/final_mesh.stl"))

		println("-----------------Finished writing results--------------------------")


		/*// Cyclic exclusive shape fitting: Just cycle over a set of variances several times.
		/*println("-------------Testing Cyclic exclusive Shape fitting-------------------------")
		var coeffs_cyclic_exclusive = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm_for_cyclic_exclusive.statisticalModel.coefficients(asm_for_cyclic_exclusive.statisticalModel.mean))
		for (outer <- 1 to 10) {
		  println("Running iteration " + outer)
		  for (i <- outer to 10) {
			println("Running iteration " + outer + " with variance " + (0.5f / (i * i)) + " and takeSize " + 500)
			coeffs_cyclic_exclusive = runShapeFittingVariable(asm_for_cyclic_exclusive, prepImg, coeffs_cyclic_exclusive, 0.5f / (i * i), 500)
			println("-----------------------------DONE--------------------------------------")
		  }
		}
		var result_cyclic_exclusive = coeffs_cyclic_exclusive.rotationParameters.toString() + "\n" + coeffs_cyclic_exclusive.translationParameters.toString() + "\n" + coeffs_cyclic_exclusive.modelCoefficients.toString()
		Files.write(Paths.get(s"exclusive_coeffs_for_$targetname.txt"), result_cyclic_exclusive.getBytes(StandardCharsets.UTF_8))
		ui.show(asm_for_cyclic_exclusive.statisticalModel, "model_updating")
		ui.setCoefficientsOf("model_updating", coeffs_cyclic_exclusive.modelCoefficients)
		val best_prob_exclusive = best_prob_ever
		best_prob_ever = 0.0
		println("-----------------Done with Cyclic exclusive Shape fitting--------------------------")*/


		// Single component fitting. Iterates over each Principal Component, fitting mostly that one component.


		// Cyclic shape fitting with Repositioning: Do normal Cyclic shape fitting, but add position fitting after each outer iteration.
		/*println("-------------Testing Cyclic Shape fitting with Repositioning-------------------------")
		var coeffs_cyclic_repositioning = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm_for_cyclic_exclusive.statisticalModel.coefficients(asm_for_cyclic_exclusive.statisticalModel.mean))
		/*for (outer <- 1 to 10) {
		  println("Running iteration " + outer)
		  for (i <- outer to 10) {
			println("Running iteration " + outer + " with variance " + (0.5f / (i * i)) + " and takeSize " + 500)
			coeffs_cyclic_repositioning = runShapeFittingVariable(asm_for_cyclic_repositioning, prepImg, coeffs_cyclic_repositioning, 0.5f / (i * i), 500)
			println("-----------------------------DONE--------------------------------------")
		  }
		  runPositionFittingVariable(asm_for_cyclic_repositioning, prepImg, coeffs_cyclic_repositioning, 0.05f, 0.05f, 20, 0)
		}*/

		var result_cyclic_repositioning = coeffs_cyclic_repositioning.rotationParameters.toString() + "\n" + coeffs_cyclic_repositioning.translationParameters.toString() + "\n" + coeffs_cyclic_repositioning.modelCoefficients.toString()
		Files.write(Paths.get(s"repositioning_coeffs_for_$targetname.txt"), result_cyclic_repositioning.getBytes(StandardCharsets.UTF_8))
		ui.show(asm_for_cyclic_repositioning.statisticalModel, "model_cyclic_repositioning")
		ui.setCoefficientsOf("model_cyclic_repositioning", coeffs_cyclic_repositioning.modelCoefficients)
		val best_prob_repositioning = best_prob_ever
		best_prob_ever = 0.0
		println("-----------------Done with Cyclic Shape fitting with Repositioning--------------------------")

		println("Best prob for exclusive method: " + best_prob_exclusive)
		println("Best prob for repositioning method: " + best_prob_repositioning)*/*/

	}


	// Sets center_of_mass for the given ASM. The idea is that center_of_mass always contains the CoM of the currently used
	// asm's MEAN MESH (i.e. is not changed by transformations&stuff)
	def calculate_CoM(asm: ActiveShapeModel): Unit = {
		var center: geometry.Vector[_3D] = Vector3D(0.0f, 0.0f, 0.0f)
		val asm_pointIds = asm.statisticalModel.mean.pointIds
		while (asm_pointIds.hasNext) {
			val next = asm.statisticalModel.mean.point(asm_pointIds.next()).toVector
			center = center + next
		}
		center_of_mass = center.map{i: Float => i/asm.statisticalModel.mean.numberOfPoints}.toBreezeVector
	}


	def runPositionFittingVariable(asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, variance_rot: Float, variance_trans: Float, takeSize: Int, use_correspondence: Int): ShapeParameters = {
		//val samples = UniformSampler(image.domain.boundingBox, 1000).sample.map(i => i._1)

		val logger = new AcceptRejectLogger[ShapeParameters] {
			private var accepted = 0f
			private var all = 0f

			override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
				accepted += 1
				all += 1

				val ratio = accepted / all
				best_prob_ever = evaluator.logValue(sample)
				println(s"!!!Accepted!!! proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
			}

			override def reject(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
				all += 1
				val ratio = accepted / all
				println(s"Rejected proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
			}
		}


		System.out.println("Running position fitting...")

		//Use_correspondence: 0 for intensity evaluator, 1 for corr.evaluator, 2 for both
		var posteriorEvaluator: ProductEvaluator[MCMC.ShapeParameters] = null

		if (use_correspondence == 0) {
			posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), IntensityBasedLikeliHoodEvaluator(asm, prepImg))
		} else if (use_correspondence == 1) {
			posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), CorespondenceBasedEvaluator(asm.statisticalModel, landmarks_on_asm zip landmarks_on_target, 1))
		} else {
			posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), IntensityBasedLikeliHoodEvaluator(asm, prepImg), CorespondenceBasedEvaluator(asm.statisticalModel, landmarks_on_asm zip landmarks_on_target, 1))
		}

		val positionGenerator = MixtureProposal.fromProposalsWithTransition((0.5, RotationUpdateProposal(variance_rot)), (0.5, TranslationUpdateProposal(variance_trans)))(rnd = new Random())

		val chain = MetropolisHastings(positionGenerator, posteriorEvaluator, logger)(new Random())
		val mhIt = chain.iterator(initialParameters)

		// Removed best prob condition. As the chain only contains accepted proposals, we have to update the model's coefficients every time,
		// and not just when the last prob was larger (i.e. a proposal may be accepted and show up here even if its prob is lower)
		val samplingIterator = for (theta <- mhIt) yield {
			val current_translation_coeffs = center_of_mass + theta.translationParameters
			val current_CoM: Point3D = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))

			//println("New chain element: "+current_translation_coeffs.toString()+", new CoM is "+current_CoM.toString())

			ui.setCoefficientsOf("model_updating", theta.modelCoefficients)

			val rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
			val rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(theta.translationParameters, theta.rotationParameters))

			val smInstance = ui.frame.scene.find[ShapeModelView](si => si.name == "model_updating").head.instances.head
			smInstance.rigidTransformation = Some(rigidtrans)
			theta
		}
		val samples = samplingIterator.drop(takeSize / 10).take(takeSize).toIndexedSeq
		samples.maxBy(posteriorEvaluator.logValue)
	}

	def runShapeFittingVariable(asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters  : ShapeParameters, variance: Float, takeSize: Int): ShapeParameters = {
	val logger = new AcceptRejectLogger[ShapeParameters] {
	  private var accepted = 0f
	  private var all = 0f

	  override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
		accepted += 1
		all += 1

		val ratio = accepted / all
		best_prob_ever = evaluator.logValue(sample)
		println(s"!!!Accepted!!! proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
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

	val chain = MetropolisHastings[ShapeParameters](poseGenerator, posteriorEvaluator, logger)(new Random())
	val mhIt = chain.iterator(initialParameters)

	val samplingIterator = for (theta <- mhIt) yield {
		ui.setCoefficientsOf("model_updating", theta.modelCoefficients)
		theta
	}

	val samples = samplingIterator.drop(takeSize / 10).take(takeSize).toIndexedSeq
	samples.maxBy(posteriorEvaluator.logValue)
	}

	def runPoseFitting(asm: ActiveShapeModel, image: DiscreteScalarImage[_3D, Float]): Unit = {
	val samples = UniformSampler(image.domain.boundingBox, 1000).sample.map(i => i._1)

	System.out.println("Running pose fitting...")

	val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), ProximityEvaluator(asm.statisticalModel, samples))

	val poseGenerator = MixtureProposal.fromProposalsWithTransition((0.2, RotationUpdateProposal(0.1f)), (0.2, TranslationUpdateProposal(0.1f)))(rnd = new Random())
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

	val poseGenerator = MixtureProposal.fromProposalsWithTransition((0.2, RotationUpdateProposal(0.1f)), (0.2, TranslationUpdateProposal(0.1f)))(rnd = new Random())
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

	def runShapeFittingForComponents(asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, tillComponentIndex: Int): ShapeParameters = {
	val logger = new AcceptRejectLogger[ShapeParameters] {
	  private var accepted = 0f
	  private var all = 0f

	  override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
		accepted += 1
		all += 1

		val ratio = accepted / all
		println(s"!!!Accepted!!! proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
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
		println(s"!!!Accepted!!! proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
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

	val chain = MetropolisHastings[ShapeParameters](poseGenerator, posteriorEvaluator, logger)(new Random())

	val mhIt = chain.iterator(initialParameters)

	val samplingIterator = for (theta <- mhIt) yield {
	  theta
	}

	val samples = samplingIterator.take(2000).toIndexedSeq

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
        println(s"!!!Accepted!!! proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
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

    val chain = new AdaptiveMetropolis[ShapeParameters](proposal, posteriorEvaluator, logger)(new Random())

    val mhIt = chain.iterator(initialParameters)

    val samplingIterator = for (theta <- mhIt) yield {
      theta
    }

    val samples = samplingIterator.take(2000).toIndexedSeq

    samples.maxBy(posteriorEvaluator.logValue)
  }
}

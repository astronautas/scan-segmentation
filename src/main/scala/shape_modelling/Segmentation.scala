package shape_modelling

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.geometry
import scalismo.geometry.{Point3D, Vector3D, _3D}
import scalismo.io.{ActiveShapeModelIO, ImageIO, MeshIO}
import scalismo.registration.{RigidTransformation, RigidTransformationSpace}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator}
import scalismo.statisticalmodel.asm.{ActiveShapeModel, PreprocessedImage}
import scalismo.ui.ShapeModelView
import scalismo.ui.api.SimpleAPI.ScalismoUI
import shape_modelling.IntensityBasedLikelyhoodEvaluators.IntensityBasedLikeliHoodEvaluatorForRigidFittingFast
import shape_modelling.MCMC._

import scala.util.Random

object Segmentation {
  var best_prob_ever = 0.0
  var center_of_mass: DenseVector[Float] = DenseVector.zeros[Float](3)
  var landmarks_on_asm: Seq[PointId] = null
  var landmarks_on_target: Seq[Point3D] = null

  var useUI: Boolean = false

  var asm: ActiveShapeModel = _
  var load_fitted_asm: Boolean = true
  private[this] var ui: ScalismoUI = _

  def main(args: Array[String]) {
    var current_CoM: Point3D = null
    var rigidTransSpace: RigidTransformationSpace[_3D] = null
    var rigidtrans: RigidTransformation[_3D] = null

    if (args.length == 0) {
      println("Provide the root path for handed data (i.e. handedData)")
      System.exit(0)
    }

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    val handedDataRootPath = args(0)
    val femurAsmFile = args(1)
    var variance_rot = args(2).toFloat
    var variance_trans = args(3).toFloat
    var pose_take_size = args(4).toInt
    var shapeTakeSize = args(5).toInt
    var shapeStDev = args(6).toFloat
    useUI = args(7).toBoolean
    
    val targetname = "37"

    // create a visualization window
    if (useUI) {
      ui = ScalismoUI()
    }

    // Read data
    asm = ActiveShapeModelIO.readActiveShapeModel(new File(s"$handedDataRootPath/$femurAsmFile")).get
    val image = ImageIO.read3DScalarImage[Short](new File(s"$handedDataRootPath/targets/$targetname.nii")).get.map(_.toFloat)

    if (useUI) {
      ui.show(asm.statisticalModel, "model_updating")
      ui.show(image, "CT")
    }

    calculate_CoM(asm)

    // PREPROCESSING
    System.out.println("Preprocessing...")
    val prepImg: PreprocessedImage = asm.preprocessor(image)




    // START POSE FITTING
    println("-------------Doing Pose fitting-------------------------")
    var coeffs = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm.statisticalModel.coefficients(asm.statisticalModel.mean))

    println("Running iteration pose fitting with variances rot/trans " + variance_rot + "/" + variance_trans + " and take_size " + pose_take_size)
    coeffs = runPoseFitting(fast = true, asm, prepImg, coeffs, variance_rot, variance_trans, pose_take_size, 0)
    coeffs = runPoseFitting(fast = false, asm, prepImg, coeffs, variance_rot, variance_trans, pose_take_size, 0)

    println("-----------------------------Saving pose fitted ASM--------------------------------------")

    // Creating the transformation according to the pose coefficients and rigid transforming the ASM
    var curr_pose_coefs = center_of_mass + coeffs.translationParameters
    current_CoM = new Point3D(curr_pose_coefs.valueAt(0), curr_pose_coefs.valueAt(1), curr_pose_coefs.valueAt(2))
    rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
    rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(coeffs.translationParameters, coeffs.rotationParameters))
    asm = asm.transform(rigidtrans)

    ActiveShapeModelIO.writeActiveShapeModel(asm, new File(s"$handedDataRootPath/femur-asm_pose_fitted.h5"))

    println("-----------------------------ASM saved--------------------------------------")
    println("-----------------------------Pose Fitting done--------------------------------------")
    // END POSE FITTING


    // START SHAPE FITTING
    println("-----------------------------Doing Shape fitting--------------------------------------")

    // We work with the pose fitted ASM now, so reset the coefficients
    if (load_fitted_asm) {
      asm = ActiveShapeModelIO.readActiveShapeModel(new File(s"$handedDataRootPath/femur-asm_pose_fitted.h5")).get
    }

    if (useUI) {
      ui.remove("model_updating")
      ui.show(asm.statisticalModel, "model_updating")
    }

    coeffs = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm.statisticalModel.coefficients(asm.statisticalModel.mean))

    // Not doing outer iteration right now. I suspect that it's actually counterproductive (time would be better spent on fine-detail fitting).
    coeffs = runShapeFitting(asm, prepImg, coeffs, shapeStDev, shapeTakeSize)
    coeffs = runPoseFitting(fast = false, asm, prepImg, coeffs, variance_rot, variance_trans, pose_take_size, 0)

    println("-----------------Shape Fitting done--------------------------")

    // Saving coefficient vectors and resulting mesh.
    val result = coeffs.rotationParameters.toString() + "\n" + coeffs.translationParameters.toString() + "\n" + coeffs.modelCoefficients.toString()
    Files.write(Paths.get(s"test_coeffs_for_$targetname.txt"), result.getBytes(StandardCharsets.UTF_8))

    var final_mesh = asm.statisticalModel.instance(coeffs.modelCoefficients)

    // This time we need to transform using the CoM of the NEW asm, i.e. of the pose_fitted_asm
    calculate_CoM(asm)

    // Figure out transform based on coeffs from the shape fitting phase. They should be all 0 right now, but if we ever introduce pose fitting stuff
    // into the shape fitting phase, this here will come in handy.
    var current_translation_coeffs = center_of_mass + coeffs.translationParameters
    current_CoM = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))
    rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
    rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(coeffs.translationParameters, coeffs.rotationParameters))

    final_mesh = final_mesh.transform(rigidtrans)
    MeshIO.writeMesh(final_mesh, new File(s"$handedDataRootPath/final_mesh.stl"))

    println("-----------------Finished writing results--------------------------")
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
    center_of_mass = center.map { i: Float => i / asm.statisticalModel.mean.numberOfPoints }.toBreezeVector
  }


  def runPoseFitting(fast: Boolean, asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, variance_rot: Float, variance_trans: Float, takeSize: Int, use_correspondence: Int): ShapeParameters = {
    //val samples = UniformSampler(image.domain.boundingBox, 1000).sample.map(i => i._1)

    val logger = new AcceptRejectLogger[ShapeParameters] {
      private var accepted = 0f
      private var all = 0f

      override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
        accepted += 1
        all += 1

        val ratio = accepted / all
        best_prob_ever = evaluator.logValue(sample)
        println(s"[$all] !!!Accepted!!! proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
      }

      override def reject(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
        all += 1
        val ratio = accepted / all
        println(s"[$all] Rejected proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
      }
    }


    System.out.println("Running position fitting...")

    //Use_correspondence: 0 for intensity evaluator, 1 for corr.evaluator, 2 for both
    var posteriorEvaluator : DistributionEvaluator[ShapeParameters] = null

    if (fast) {
      posteriorEvaluator = IntensityBasedLikeliHoodEvaluatorForRigidFittingFast(asm, prepImg)
    } else {
      posteriorEvaluator = IntensityBasedLikelyhoodEvaluators.IntensityBasedLikeliHoodEvaluatorForRigidFitting(asm, prepImg)
    }

    val positionGenerator = MixtureProposal.fromProposalsWithTransition((0.1, RotationUpdateProposal(variance_rot * 4)), (0.4, RotationUpdateProposal(variance_rot)), (0.4, TranslationUpdateProposal(variance_trans)), (0.1, TranslationUpdateProposal(variance_trans * 4)))(rnd = new Random())

    val chain = MetropolisHastings(positionGenerator, posteriorEvaluator, logger)(new Random())
    val mhIt = chain.iterator(initialParameters)

    // Removed best prob condition. As the chain only contains accepted proposals, we have to update the model's coefficients every time,
    // and not just when the last prob was larger (i.e. a proposal may be accepted and show up here even if its prob is lower)
    val samplingIterator = for (theta <- mhIt) yield {

      if (useUI) {
        val current_translation_coeffs = center_of_mass + theta.translationParameters
        val current_CoM: Point3D = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))

        val rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
        val rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(theta.translationParameters, theta.rotationParameters))

        val smInstance = ui.frame.scene.find[ShapeModelView](si => si.name == "model_updating").head.instances.head
        smInstance.rigidTransformation = Some(rigidtrans)
      }

      theta
    }

    val samples = samplingIterator.drop(takeSize / 10).take(takeSize).toIndexedSeq
    samples.maxBy(posteriorEvaluator.logValue)
  }

  def runShapeFitting(asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, variance: Float, takeSize: Int): ShapeParameters = {
    val logger = new AcceptRejectLogger[ShapeParameters] {
      private var accepted = 0f
      private var all = 0f

      override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
        accepted += 1
        all += 1

        val ratio = accepted / all
        best_prob_ever = evaluator.logValue(sample)
        println(s"[$all]!!!Accepted!!! proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
      }

      override def reject(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
        all += 1
        val ratio = accepted / all
        println(s"[$all]Rejected proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
      }
    }

    System.out.println("Running runShapeFitting...")

    val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), IntensityBasedLikelyhoodEvaluators.IntensityBasedLikeliHoodEvaluatorForShapeFitting(asm, prepImg))
    // Deviations should match deviations of model
    val poseGenerator = MixtureProposal.fromProposalsWithTransition((1.0, ShapeUpdateProposal(asm.statisticalModel.rank, variance)))(rnd = new Random())

    val chain = MetropolisHastings[ShapeParameters](poseGenerator, posteriorEvaluator, logger)(new Random())
    val mhIt = chain.iterator(initialParameters)

    val samplingIterator = for (theta <- mhIt) yield {

      if (useUI) {
        ui.setCoefficientsOf("model_updating", theta.modelCoefficients)
      }

      theta
    }

    val samples = samplingIterator.drop(takeSize / 10).take(takeSize).toIndexedSeq
    samples.maxBy(posteriorEvaluator.logValue)
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

    val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), IntensityBasedLikelyhoodEvaluators.IntensityBasedLikeliHoodEvaluatorForShapeFitting(asm, prepImg))

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

    val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), IntensityBasedLikelyhoodEvaluators.IntensityBasedLikeliHoodEvaluatorForShapeFitting(asm, prepImg))

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
}

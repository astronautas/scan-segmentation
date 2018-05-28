package shape_modelling

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.text.SimpleDateFormat

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.geometry
import scalismo.geometry._
import scalismo.io.{ActiveShapeModelIO, ImageIO, LandmarkIO, MeshIO}
import scalismo.registration.{RigidTransformation, RigidTransformationSpace}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator}
import scalismo.statisticalmodel.asm.{ActiveShapeModel, PreprocessedImage}
import scalismo.ui.ShapeModelView
import scalismo.ui.api.SimpleAPI.ScalismoUI
import shape_modelling.Evaluators.Pose_aware_evaluator_fast
import shape_modelling.MCMC._

import scala.collection.mutable.ListBuffer
import scala.util.Random

object Segmentation {
  val time = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
  var best_prob_ever = 0.0
  var center_of_mass: DenseVector[Float] = DenseVector.zeros[Float](3)
  var landmarks_on_asm: Seq[PointId] = null
  var landmarks_on_target: Seq[Point3D] = null
  var useUI: Boolean = false
  var asm: ActiveShapeModel = _
  var load_fitted_asm: Boolean = false
  var plotter: HastingsPlotter = _
  var graph: Boolean = false
  var allIts = 0
  var asm_lms: Seq[Landmark[_3D]] = _
  var target_lms: Seq[Landmark[_3D]] = _
  private[this] var ui: ScalismoUI = _

  def main(args: Array[String]) {
    var current_CoM: Point3D = null
    var rigidTransSpace: RigidTransformationSpace[_3D] = null
    var rigidtrans: RigidTransformation[_3D] = null

    if (args.length != 11) {
      println("args should be of form: <handedData_path><femur_asm_file><variance_rot><variance_trans><pose_take_size><shape_take_size><shape_variance><use_ui(true/false)><path_to_target><variance_scaling_factor><plot_graph>")
      println("Remember, path to target needs to be preceded by either targets/ or test/. Variance scaling factor is used to change variance decrement during shape fitting")
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
    var targetname = args(8)
    var decayParam = args(9).toFloat
    graph = args(10).toBoolean

    //val targetname = "4"

    if (graph) {
      plotter = new HastingsPlotter(frequency = 1)
    }

    asm_lms = LandmarkIO.readLandmarksJson[_3D](new File(s"$handedDataRootPath/femur-landmarks.json")).get
    target_lms = LandmarkIO.readLandmarksJson[_3D](new File(s"$handedDataRootPath/$targetname.json")).get

    // Read data
    asm = ActiveShapeModelIO.readActiveShapeModel(new File(s"$handedDataRootPath/$femurAsmFile")).get
    val image = ImageIO.read3DScalarImage[Short](new File(s"$handedDataRootPath/$targetname.nii")).get.map(_.toFloat)

    if (useUI) {
      ui = ScalismoUI()
      ui.show(asm.statisticalModel, "model_updating")
      ui.show(image, "CT")
      if (targetname.split("/")(0) == "test") {
        ui.show(MeshIO.readMesh(new File(s"$handedDataRootPath/$targetname.stl")).get, "reference femur mesh")
      }
      ui.addLandmarksTo(asm_lms, "model_updating")
      ui.addLandmarksTo(target_lms, "CT")
      asm_lms = ui.getLandmarksOf("model_updating").get
      target_lms = ui.getLandmarksOf("CT").get
    }

    calculate_CoM(asm)

    // PREPROCESSING
    System.out.println("Preprocessing...")
    val prepImg: PreprocessedImage = asm.preprocessor(image)

    // START POSE FITTING
    println("-------------Doing Pose fitting-------------------------")
    var coeffs = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm.statisticalModel.coefficients(asm.statisticalModel.mean))
    //coeffs = runPoseFitting(fast = true, asm, prepImg, coeffs, variance_rot, variance_trans, pose_take_size*10, 0)

    println("Running pose fitting with variances rot/trans " + variance_rot + "/" + variance_trans + " and take_size " + pose_take_size)

    // Run initial pose fitting using landmark evaluator.
    coeffs = runPoseFitting(use_landmarks = true, asm, prepImg, coeffs, variance_rot, variance_trans, pose_take_size * 3, 0)

    if (useUI) {
      transformModelOnUI(coeffs)
    }

    println("-----------------------------Saving pose fitted ASM--------------------------------------")

    ActiveShapeModelIO.writeActiveShapeModel(asm, new File(s"$handedDataRootPath/femur-asm_pose_fitted.h5"))

    println("-----------------------------ASM saved--------------------------------------")
    println("-----------------------------Pose Fitting done--------------------------------------")
    // END POSE FITTING

    // START SHAPE FITTING
    println("-----------------------------Doing Shape fitting--------------------------------------")

    // We work with the pose fitted ASM now, so reset the coefficients
    //if (load_fitted_asm) {
    //  asm = ActiveShapeModelIO.readActiveShapeModel(new File(s"$handedDataRootPath/femur-asm_pose_fitted.h5")).get
    //}
    // For single version, set i<-1 to 1 and give large enough take_size as program arguments.
    for (i <- 1 to 3) {
      var rotSt = (variance_rot.toDouble / (10 * i * decayParam.toDouble)).toFloat
      var transSt = (variance_trans.toDouble / (10 * i * decayParam.toDouble)).toFloat
      val variance = (shapeStDev.toDouble / (i * i * decayParam.toDouble)).toFloat

      println(s"stDevRot: $rotSt, stDevTrans: $transSt")

      coeffs = runShapeFitting(asm, prepImg, coeffs, variance, shapeTakeSize)
      transformModelOnUI(coeffs)
      coeffs = runPoseFitting(use_landmarks = false, asm, prepImg, coeffs, rotSt, transSt, pose_take_size, 0)

      //Between pose fitting and shape fitting, we need to apply the transform coefficients. LMs are ignored at this point
      // (they're only used for the initial fitting, as they're not very accurate)

      if (useUI) {
        transformModelOnUI(coeffs)
      }
    }

    //coeffs = ShapeParameters(DenseVector.zeros[Float](3), DenseVector.zeros[Float](3), asm.statisticalModel.coefficients(asm.statisticalModel.mean))

    println("-----------------Shape Fitting done--------------------------")
    coeffs = transformModel(coeffs)._1

    // Saving coefficient vectors and resulting mesh.
    var result = coeffs.rotationParameters.toString() + "\n" + coeffs.translationParameters.toString() + "\n" + coeffs.modelCoefficients.toString() + "\n"
    targetname = targetname.split("/")(1)
    for (i <- 0 to args.length - 1) {
      result += args(i).toString + " "

    }
    Files.write(Paths.get(s"test_coeffs_for_$targetname.txt"), result.getBytes(StandardCharsets.UTF_8))
    var final_mesh = asm.statisticalModel.instance(coeffs.modelCoefficients)

    //This seems to be unnecessary now, as after each shape fitting iteration, the model is transformed.
    /* This time we need to transform using the CoM of the NEW asm, i.e. of the pose_fitted_asm
    //calculate_CoM(asm)

    // Figure out transform based on coeffs from the shape fitting phase. They should be all 0 right now, but if we ever introduce pose fitting stuff
    // into the shape fitting phase, this here will come in handy.
    var current_translation_coeffs = center_of_mass + coeffs.translationParameters
    current_CoM = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))
    rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
    rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(coeffs.translationParameters, coeffs.rotationParameters))

    final_mesh = final_mesh.transform(rigidtrans)*/



    MeshIO.writeMesh(final_mesh, new File(s"$handedDataRootPath/final_mesh_$targetname.stl"))

    println("-----------------Finished writing results--------------------------")
  }

  def transformModelOnUI(coeffs: ShapeParameters) : Unit = {
    ui.remove("model_updating")
    ui.show(asm.statisticalModel, "model_updating")
    ui.setCoefficientsOf("model_updating", coeffs.modelCoefficients)

    val current_translation_coeffs = center_of_mass + coeffs.translationParameters
    val current_CoM: Point3D = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))

    val rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
    val rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(coeffs.translationParameters, coeffs.rotationParameters))

    val smInstance = ui.frame.scene.find[ShapeModelView](si => si.name == "model_updating").head.instances.head
    smInstance.rigidTransformation = Some(rigidtrans)
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

  def runPoseFitting(use_landmarks: Boolean, asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, variance_rot: Float, variance_trans: Float, takeSize: Int, use_correspondence: Int): ShapeParameters = {
    //val samples = UniformSampler(image.domain.boundingBox, 1000).sample.map(i => i._1)

    val logger = new AcceptRejectLogger[ShapeParameters] {
      private var accepted = 0f
      private var all = 0f

      override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {

        if (graph) {
          plotter.offer(allIts.toDouble, evaluator.logValue(sample))
        }

        accepted += 1
        all += 1
        allIts += 1

        val ratio = accepted / all
        best_prob_ever = evaluator.logValue(sample)
        println(s"[$all] !!!Accepted!!! proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
      }

      override def reject(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
        allIts += 1

        all += 1
        val ratio = accepted / all
        println(s"[$all] Rejected proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
      }
    }


    System.out.println("Running position fitting...")

    //Use_correspondence: 0 for intensity evaluator, 1 for corr.evaluator, 2 for both
    var posteriorEvaluator: DistributionEvaluator[ShapeParameters] = null

    if (use_landmarks) {
      val modelLmIds = asm_lms.map(l => asm.statisticalModel.mean.pointId(l.point).get)
      val targetPoints = target_lms.map(l => l.point)
      val correspondences = modelLmIds.zip(targetPoints)

      posteriorEvaluator = CorespondenceBasedEvaluator(asm.statisticalModel, correspondences, 1.0)
    } else {
      posteriorEvaluator = Evaluators.Pose_aware_evaluator(asm, prepImg)
    }

    val positionGenerator = MixtureProposal.fromProposalsWithTransition((0.1, RotationUpdateProposal(variance_rot * 10)), (0.4, RotationUpdateProposal(variance_rot)), (0.4, TranslationUpdateProposal(variance_trans)), (0.1, TranslationUpdateProposal(variance_trans * 5)))(rnd = new Random())

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
    val max = samples.maxBy(posteriorEvaluator.logValue)
    val maxVal = posteriorEvaluator.logValue(max)

    println(s"MAX theta: $maxVal")
    best_prob_ever = maxVal
    max
  }

  def runShapeFitting(asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, variance: Float, takeSize: Int): ShapeParameters = {
    val logger = new AcceptRejectLogger[ShapeParameters] {
      private var accepted = 0f
      private var all = 0f

      override def accept(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
        if (graph) {
          plotter.offer(allIts.toDouble, evaluator.logValue(sample))
        }

        accepted += 1
        all += 1
        allIts += 1

        val ratio = accepted / all
        best_prob_ever = evaluator.logValue(sample)
        println(s"[$all]!!!Accepted!!! proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
      }

      override def reject(current: ShapeParameters, sample: ShapeParameters, generator: ProposalGenerator[ShapeParameters], evaluator: DistributionEvaluator[ShapeParameters]): Unit = {
        all += 1
        allIts += 1

        val ratio = accepted / all
        println(s"[$all]Rejected proposal generated by $generator (probability ${evaluator.logValue(sample)}) : $ratio")
      }
    }

    System.out.println("Running runShapeFitting...")

    val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), Evaluators.Pose_aware_evaluator(asm, prepImg))
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
    val max = samples.maxBy(posteriorEvaluator.logValue)
    val maxVal = posteriorEvaluator.logValue(max)

    println(s"MAX theta: $maxVal")
    best_prob_ever = maxVal
    max
  }

  def transformModel(coeffs: ShapeParameters, landmarks: Seq[Landmark[_3D]] = null): (ShapeParameters, Seq[Landmark[_3D]]) = {
    var translated_CoM_vector = center_of_mass + coeffs.translationParameters
    var current_CoM = new Point3D(translated_CoM_vector.valueAt(0), translated_CoM_vector.valueAt(1), translated_CoM_vector.valueAt(2))
    var rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
    var rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(coeffs.translationParameters, coeffs.rotationParameters))

    asm = asm.transform(rigidtrans)
    center_of_mass = translated_CoM_vector

    val newLms: ListBuffer[Landmark[_3D]] = ListBuffer()

    if (landmarks != null) {
      landmarks.zipWithIndex foreach { case (landmark, i) =>
        val point = rigidtrans.f(landmark.point)
        val lm = Landmark(landmark.id, point = point)
        newLms += lm
      }
    }

    return (ShapeParameters(DenseVector.zeros(3), DenseVector.zeros(3), coeffs.modelCoefficients), newLms)
  }

  def runPoseFittingOnlyRotationAlongX(fast: Boolean, asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, variance_rot: Float, variance_trans: Float, takeSize: Int, use_correspondence: Int): ShapeParameters = {
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
    var posteriorEvaluator: DistributionEvaluator[ShapeParameters] = null

    if (fast) {
      posteriorEvaluator = Pose_aware_evaluator_fast(asm, prepImg)
    } else {
      posteriorEvaluator = Evaluators.Pose_aware_evaluator(asm, prepImg)
    }

    val positionGenerator = MixtureProposal.fromProposalsWithTransition((0.8, RotationUpdateProposalX(variance_rot)), (0.2, RotationUpdateProposalX(variance_rot * 2)))(rnd = new Random())

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

  def runPoseFittingOnlyTranslationAllAxis(fast: Boolean, asm: ActiveShapeModel, prepImg: PreprocessedImage, initialParameters: ShapeParameters, variance_rot: Float, variance_trans: Float, takeSize: Int, use_correspondence: Int): ShapeParameters = {
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
    var posteriorEvaluator: DistributionEvaluator[ShapeParameters] = null

    if (fast) {
      posteriorEvaluator = Pose_aware_evaluator_fast(asm, prepImg)
    } else {
      posteriorEvaluator = Evaluators.Pose_aware_evaluator(asm, prepImg)
    }

    val positionGenerator = MixtureProposal.fromProposalsWithTransition((0.8, TranslationUpdateProposal(variance_trans)), (0.2, TranslationUpdateProposal(variance_trans * 4)))(rnd = new Random())

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

    val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), Evaluators.Shape_evaluator(asm, prepImg))

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

    val posteriorEvaluator = ProductEvaluator(MCMC.ShapePriorEvaluator(asm.statisticalModel), Evaluators.Shape_evaluator(asm, prepImg))

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

package shape_modelling

import breeze.linalg.DenseVector
import scalismo.geometry.{Point3D, _3D}
import scalismo.mesh.TriangleMesh
import scalismo.registration.RigidTransformationSpace
import scalismo.statisticalmodel.asm.{ActiveShapeModel, PreprocessedImage}
import shape_modelling.MCMC.ShapeParameters

import scala.collection.mutable.ArrayBuffer

object LikelihoodChecker {

  private final val OUT_OF_BOUNDS_PROBABILITY = -400000.0

  // Mesh has to be in correspondence with asm
  def likelihoodThatMeshFitsImage(asm: ActiveShapeModel, theta: ShapeParameters, preprocessedImage: PreprocessedImage): Double = {

    // Need to consider rot/trans as well, so we first apply the transform to the mesh here.
    var mesh = asm.statisticalModel.instance(theta.modelCoefficients)
    val current_translation_coeffs = Segmentation.center_of_mass + theta.translationParameters
    val current_CoM: Point3D = new Point3D(current_translation_coeffs.valueAt(0), current_translation_coeffs.valueAt(1), current_translation_coeffs.valueAt(2))

    val rigidTransSpace = RigidTransformationSpace[_3D](current_CoM)
    val rigidtrans = rigidTransSpace.transformForParameters(DenseVector.vertcat(theta.translationParameters, theta.rotationParameters))

	mesh = mesh.transform(rigidtrans)

    val ids = asm.profiles.ids
    var parList = ArrayBuffer.fill(ids.length)(OUT_OF_BOUNDS_PROBABILITY)

    asm.profiles.ids.par.zipWithIndex foreach { case (id, i) =>
      val profile = asm.profiles(id)
      val profilePointOnMesh = mesh.point(profile.pointId)

      // TODO - sometimes featureExtractor returns None
      val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh, profile.pointId).orNull

      // Having a profile point, we check whether a point with same id on mesh has "same" intensity
      // if so, mesh point with same id overlaps with profile point of model, meaning mesh point is in correct position of a image (is "fit")
      // returns probability of this

      if (featureAtPoint != null) {
        parList(i) = profile.distribution.logpdf(featureAtPoint)
      }
    }

    parList.sum
  }

  def likelihoodThatMeshFitsImageNonParallel(asm: ActiveShapeModel, mesh: TriangleMesh, preprocessedImage: PreprocessedImage): Double = {

    val ids = asm.profiles.ids

    val likelihoods = for (id <- ids) yield {
      val profile = asm.profiles(id)
      val profilePointOnMesh = mesh.point(profile.pointId)

      // TODO - sometimes featureExtractor returns None
      val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh, profile.pointId).orNull

      // Having a profile point, we check whether a point with same id on mesh has "same" intensity
      // if so, mesh point with same id overlaps with profile point of model, meaning mesh point is in correct position of a image (is "fit")
      // returns probability of this

      if (featureAtPoint != null) {
        profile.distribution.logpdf(featureAtPoint)
      } else {
        OUT_OF_BOUNDS_PROBABILITY
      }
    }

    likelihoods.sum
  }
}

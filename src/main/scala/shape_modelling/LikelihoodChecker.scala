package shape_modelling

import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.asm.{ActiveShapeModel, PreprocessedImage}

object LikelihoodChecker {

  // Mesh has to be in correspondence with asm
  def likelihoodThatMeshFitsImage(asm: ActiveShapeModel, mesh: TriangleMesh, preprocessedImage: PreprocessedImage): Double = {

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
        0
      }
    }

    // Point fitting is independent, just return probability sum (~ the higher probability, the better)
    likelihoods.par.sum
  }
}

package shape_modelling

import scalismo.geometry
import scalismo.geometry.{Point, _3D}
import scalismo.mesh.TriangleMesh

object Utils {
//  def centerOfMass(points : Seq[Point[_3D]]) : Point[_3D] = {
////    var sum: Point[_3D] = new Point[_3D]
////
////    sum(0) = sum(0) / points.length
////    sum(1) = sum(0) / points.length
////    sum(2) = sum(0) / points.length
////
////    sum
//  }

  def averageDistance(origin: TriangleMesh, target: TriangleMesh): Unit = {
    var sum = 0f

    origin.pointIds.foreach(id => {
      val distVector: geometry.Vector[_3D] = target.point(id) - origin.point(id)
      sum = distVector(0) + distVector(1) + distVector(2)
    })

    sum / origin.pointIds.length
  }
}

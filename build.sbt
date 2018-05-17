organization  := "ch.unibas.cs.gravis"

name := """minimal-scalismo-seed"""
version       := "0.5"

scalaVersion  := "2.11.7"

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

resolvers += Resolver.bintrayRepo("unibas-gravis", "maven")

resolvers += Opts.resolver.sonatypeSnapshots

resolvers ++= Seq(
	"scalismo" at "http://shapemodelling.cs.unibas.ch/repository/public"
)

libraryDependencies  ++= {
	Seq(
        "ch.unibas.cs.gravis" % "scalismo-ui_2.11" % "0.6.+",
        "ch.unibas.cs.gravis" % "scalismo-sampling_2.11" % "develop-SNAPSHOT"
	)
}

libraryDependencies += "org.scalanlp" % "breeze_2.11" % "0.12"
resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
scalaVersion := "2.11.6"

assemblyJarName in assembly := "exectuable.jar"

mainClass in assembly := Some("com.example.ExampleApp")


assemblyMergeStrategy in assembly <<= (assemblyMergeStrategy in assembly) { (old) =>
  {
    case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
    case PathList("META-INF", s) if s.endsWith(".SF") || s.endsWith(".DSA") || s.endsWith(".RSA") => MergeStrategy.discard
    case "reference.conf" => MergeStrategy.concat
    case _ => MergeStrategy.first
  }
}

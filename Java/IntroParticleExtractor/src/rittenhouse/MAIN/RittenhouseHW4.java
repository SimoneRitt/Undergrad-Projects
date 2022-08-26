package rittenhouse.MAIN;

import java.util.Random;
import rittenhouse.COLLECTOR.ParticleCollector;
import rittenhouse.EXTRACTOR.ParticleExtractor;
import rittenhouse.GENERATOR.ParticleGenerator;
import rittenhouse.PARTICLE.Particle;
import rittenhouse.PARTICLE.Location;
import rittenhouse.REFINER.ParticleRefiner;

public class RittenhouseHW4 {

	public static void main(String[] args) {
		
		ParticleGenerator PGen = new ParticleGenerator("PGEN-1", new Location(10,10));
		ParticleCollector PCol = new ParticleCollector("PCOL-1", new Location(12,12));
		ParticleExtractor PExt = new ParticleExtractor("PEXT-1", new Location(14,14));
		ParticleRefiner PRef = new ParticleRefiner("PREF-1", "Refinery A", new Location(16,16));
		
		
		// Generate and Collect Particles
		
		for(int i = 0; i < 100; i ++) {
			Random R1 = new Random();
			String[] M = {"Vibranium", "Unobtanium", "Lockanium", "Learacite"};
			Random R2 = new Random();
			Particle P = PGen.generateParticle(R1.nextInt(10) + 1, M[R2.nextInt(4)]);
			PCol.collectParticles(P);
		}
		
		// Verify Particles Generated and Collected
		
		System.out.println(ParticleGenerator.GENERATED_COUNT + " particles have been generated");
		System.out.println(PCol.getCollectorPool().size() + " particles have been collected");
		System.out.println();
		
		// Extract Particles and Verify Extraction
		
		PExt.extractParticles(PCol);
		System.out.println(PExt.getExtractorPool().size() + " have been extracted");
		System.out.println(PCol.getCollectorPool().size() + " particles remain in Collector");
		System.out.println();
		
		// Refine Particles
		
		PRef.emptyExtractor(PExt);
		System.out.println(PRef.getpStorage().size() + " particles are being refined");
		for(Particle P: PRef.getRefined_Storage()) {
			PRef.refineParticle(P);
		}
		System.out.println();
		
		// Sample Refined Particles
		PRef.sampleParticles(3);

	}

}

package rittenhouse.GENERATOR;

import rittenhouse.PARTICLE.Location;
import rittenhouse.PARTICLE.Particle;

public class ParticleGenerator {
	
	// variables
	
	private String pGeneratorID;
	private Location pGeneratorPos;
	
	public static int GENERATED_COUNT = 0;
	
	// constructor
	
	public ParticleGenerator(String pGeneratorID, Location pGeneratorPos) {
		this.pGeneratorID = pGeneratorID;
		this.pGeneratorPos = pGeneratorPos;
	}
	
	// getters and setters
	
	public String getpGeneratorID() {
		return pGeneratorID;
	}

	public void setpGeneratorID(String pGeneratorID) {
		this.pGeneratorID = pGeneratorID;
	}

	public Location getpGeneratorPos() {
		return pGeneratorPos;
	}

	public void setpGeneratorPos(Location pGeneratorPos) {
		this.pGeneratorPos = pGeneratorPos;
	}
	
	// methods
	
	public Particle generateParticle(double pDiameter, String pComposition) {
		
		Particle P = new Particle(pDiameter, pComposition);
		
		GENERATED_COUNT += 1;
		
		return P;
	}

}

package rittenhouse.EXTRACTOR;

import java.util.ArrayList;

import rittenhouse.COLLECTOR.ParticleCollector;
import rittenhouse.PARTICLE.Location;
import rittenhouse.PARTICLE.Particle;

public class ParticleExtractor {
	
	// variables 
	
	private String extractorID;
	private Location extractorPOS;
	private ArrayList<Particle> extractorPool;
	
	public static final int EXTRACTOR_CAPACITY = 500;
	
	// constructor
	
	public ParticleExtractor(String extractorID, Location extractorPOS) {
		this.extractorID = extractorID;
		this.extractorPOS = extractorPOS;
		this.extractorPool = new ArrayList<Particle>();
	}
	
	// getters and setters
	
	public String getExtractorID() {
		return extractorID;
	}

	public void setExtractorID(String extractorID) {
		this.extractorID = extractorID;
	}

	public Location getExtractorPOS() {
		return extractorPOS;
	}

	public void setExtractorPOS(Location extractorPOS) {
		this.extractorPOS = extractorPOS;
	}

	public ArrayList<Particle> getExtractorPool() {
		return extractorPool;
	}

	public void setExtractorPool(ArrayList<Particle> extractorPool) {
		this.extractorPool = extractorPool;
	}
	
	// methods
	
	public boolean extractParticles(ParticleCollector pc) {
		if((pc.getCollectorPool().size() + this.extractorPool.size()) <= EXTRACTOR_CAPACITY) {
			this.extractorPool.addAll(pc.getCollectorPool());
			pc.getCollectorPool().clear();
			return true;
		}
		else {
			System.out.println("Extractor Pool does not have sufficient capacity");
			return false;
		}
	}
	
}

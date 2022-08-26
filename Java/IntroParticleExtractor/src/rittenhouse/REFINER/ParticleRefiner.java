package rittenhouse.REFINER;

import java.util.ArrayList;

import rittenhouse.EXTRACTOR.ParticleExtractor;
import rittenhouse.PARTICLE.Location;
import rittenhouse.PARTICLE.Particle;

public class ParticleRefiner {
	
	// variables
	
	private String refinerID;
	private String refinerName;
	private Location refinerPos;
	private ArrayList<Particle> pStorage;
	private ArrayList<Particle> refined_Storage;
	
	// constructor
	
	public ParticleRefiner(String refinerID, String refinerName, Location refinerPos) {
		this.refinerID = refinerID;
		this.refinerName = refinerName;
		this.refinerPos = refinerPos;
		this.pStorage = new ArrayList<Particle>();
		this.refined_Storage = new ArrayList<Particle>();
	}

	// getters and setters
	
	public String getRefinerID() {
		return refinerID;
	}

	public void setRefinerID(String refinerID) {
		this.refinerID = refinerID;
	}
	
	public String getRefinerName() {
		return refinerName;
	}

	public void setRefinerName(String refinerName) {
		this.refinerName = refinerName;
	}

	public Location getRefinerPos() {
		return refinerPos;
	}

	public void setRefinerPos(Location refinerPos) {
		this.refinerPos = refinerPos;
	}

	public ArrayList<Particle> getpStorage() {
		return pStorage;
	}

	public void setpStorage(ArrayList<Particle> pStorage) {
		this.pStorage = pStorage;
	}

	public ArrayList<Particle> getRefined_Storage() {
		return refined_Storage;
	}

	public void setRefined_Storage(ArrayList<Particle> refined_Storage) {
		this.refined_Storage = refined_Storage;
	}
	
	// methods
	
	public Particle refineParticle(Particle p) {
		p.setRefined(true);
		return p;
	}
	
	public boolean emptyExtractor(ParticleExtractor pe) {
		this.pStorage.addAll(pe.getExtractorPool()); // adding Particle objects from ParticleExtractor
		
		for(int x = 0; x < this.pStorage.size(); x ++) { // copying pStorage Particles into refined_Storage
			this.refined_Storage.add(this.pStorage.get(x));
		}
		
		if(this.refined_Storage.equals(this.pStorage)) { // checking if operation was successful
			return true;
		}
		else {
			return false;
		}
	}
	
	public void sampleParticles(int count) {
		for(int x = 0; x < count; x ++) {
			this.refined_Storage.get(x).displayParticleInfo();
		}
	}
	
	public void displayRefinerInfo() {
		System.out.println("Refiner " + this.refinerID);
		System.out.println("Name: " + this.refinerName);
		System.out.println("Position: " + this.refinerPos.toString());
		System.out.println("There are " + this.getpStorage().size() + " Particles in this Refiner's storage");
		System.out.println("********************");
		
	}
	
}

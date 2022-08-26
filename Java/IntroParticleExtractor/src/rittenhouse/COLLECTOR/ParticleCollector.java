package rittenhouse.COLLECTOR;

import java.util.ArrayList;
import rittenhouse.PARTICLE.Location;
import rittenhouse.PARTICLE.Particle;

public class ParticleCollector {
	
	// variables
	
	private String collectorID;
	private Location collectorPOS;
	private ArrayList<Particle> collectorPool;
	
	public static final int COLLECTOR_CAPACITY = 500;
	
	// constructor
	
	public ParticleCollector(String collectorID, Location collectorPOS) {
		this.collectorID = collectorID;
		this.collectorPOS = collectorPOS;
		this.collectorPool = new ArrayList<Particle>();
		
	}
	
	// getters and setters
	
	public String getCollectorID() {
		return collectorID;
	}

	public void setCollectorID(String collectorID) {
		this.collectorID = collectorID;
	}

	public Location getCollectorPOS() {
		return collectorPOS;
	}

	public void setCollectorPOS(Location collectorPOS) {
		this.collectorPOS = collectorPOS;
	}

	public ArrayList<Particle> getCollectorPool() {
		return collectorPool;
	}

	public void setCollectorPool(ArrayList<Particle> collectorPool) {
		this.collectorPool = collectorPool;
	}
	
	// methods
	
	public boolean collectParticles(Particle p) {
		if((this.collectorPool.size() + 1) <= COLLECTOR_CAPACITY) {
			this.collectorPool.add(p);
			return true;
		}
		else {
			return false;
		}
	}

}

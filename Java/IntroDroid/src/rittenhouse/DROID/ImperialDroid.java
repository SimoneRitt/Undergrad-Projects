package rittenhouse.DROID;

import java.util.TreeMap;

import rittenhouse.ARMS.Arm;
import rittenhouse.ARMS.LeftArm;
import rittenhouse.ARMS.RightArm;
import rittenhouse.BODIES.Cranial;
import rittenhouse.BODIES.Torso;
import rittenhouse.INTERFACES.DroidActions;
import rittenhouse.INTERFACES.DroidOperations;
import rittenhouse.LEGS.LeftLeg;
import rittenhouse.LEGS.Leg;
import rittenhouse.LEGS.RightLeg;

public abstract class ImperialDroid implements DroidActions, DroidOperations {
	
	private String DroidID;
	private String DroidType;
	private Cranial Head;
	private Torso UpperTorso;
	private TreeMap<String, Arm> Arms;
	private TreeMap<String, Leg> Legs;
	
	public ImperialDroid(String DroidID, String DroidType) {
		this.DroidID = DroidID;
		this.DroidType = DroidType;
		this.Head = new Cranial("x");
		this.UpperTorso = new Torso("x");
		
		this.Arms = new TreeMap<String, Arm>();
		
		this.Arms.put("Left Arm", new LeftArm("x"));
		this.Arms.put("Right Arm", new RightArm("x"));
		
		this.Legs = new TreeMap<String, Leg>();
		
		this.Legs.put("Left Leg", new LeftLeg("x"));
		this.Legs.put("Right Leg", new RightLeg("x"));
	}

	public String getDroidID() {
		return DroidID;
	}

	public void setDroidID(String droidID) {
		DroidID = droidID;
	}

	public String getDroidType() {
		return DroidType;
	}

	public void setDroidType(String droidType) {
		DroidType = droidType;
	}

	public Cranial getHead() {
		return Head;
	}

	public void setHead(Cranial head) {
		Head = head;
	}

	public Torso getUpperTorso() {
		return UpperTorso;
	}

	public void setUpperTorso(Torso upperTorso) {
		UpperTorso = upperTorso;
	}

	public TreeMap<String, Arm> getArms() {
		return Arms;
	}

	public void setArms(TreeMap<String, Arm> arms) {
		Arms = arms;
	}

	public TreeMap<String, Leg> getLegs() {
		return Legs;
	}

	public void setLegs(TreeMap<String, Leg> legs) {
		Legs = legs;
	}
	
	public void displayInfo() {
		System.out.println("Droid ID: " + this.DroidID);
		System.out.println("Droid Type: " + this.DroidType);
		System.out.println("******");
	}
	
	public void runDiagnostic() {
		displayInfo();
		boolean broken = false;
		
		String[] assessment = new String[] {"Normal", "Normal", "Normal", "Normal", "Normal", "Normal"};
		
		if(this.Head.getOperational() == false) {
			broken = true;
			assessment[0] = "ERROR!!";
		}
		if(this.UpperTorso.getOperational() == false) {
			broken = true;
			assessment[1] = "ERROR!!";
		}
		if(((LeftArm)this.Arms.get("Left Arm")).getOperational() == false) {
			broken = true;
			assessment[2] = "ERROR!!";
		}
		if(((RightArm)this.Arms.get("Right Arm")).getOperational() == false) {
			broken = true;
			assessment[3] = "ERROR!!";
		}
		if(((LeftLeg)this.Legs.get("Left Leg")).getOperational() == false) {
			broken = true;
			assessment[4] = "ERROR!!";
		}
		if(((RightLeg)this.Legs.get("Right Leg")).getOperational() == false) {
			broken = true;
			assessment[5] = "ERROR!!";
		}
		
		if(broken) {
			System.out.println("Droid Has Malfunctioning Component(s)");
		}
		else {
			System.out.println("Droid Operating Within Normal Parameters");
		}
		
		System.out.println("\t\tComponent Status");
		System.out.println("HEAD Check:Head " + assessment[0]);
		System.out.println("UPPER TORSO Check:Upper Torso " + assessment[1]);
		System.out.println("ARMS Check:Left Arm " + assessment[2] + " Right Arm " + assessment[3]);
		System.out.println("LEGS Check:Left Leg " + assessment[4] + " Right Leg " + assessment[5]);
		
	}

}

using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using VehicleComponents.Actuators;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using DefaultNamespace.LookUpTable;

using DefaultNamespace; // ResetArticulationBody() extension
using System.IO; // To logged the reward data 


using Force;
using Unity.Mathematics;
using System;
using System.Collections.Generic;
using MathNet;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using Unity.Robotics.Core; //Clock
using Unity.Robotics.ROSTCPConnector;
using StdMessages = RosMessageTypes.Std;

using Random = UnityEngine.Random;

public class DroneAgent2 : Agent
{
    [Header("Basics")] 
    public GameObject BaseLink;
    public GameObject BaseLinkSAM;
    public Transform goal;             // where the drone should go 
    public DroneController.DroneController droneController;
    public float maxSpeed = 1.5f;   
    public bool debugMode = false;    

    private ArticulationBody baseLinkDroneAB;
    private ArticulationBody[] ABparts;
    private Rigidbody[] RBparts;

    [Tooltip("Transform to Teleport(Drone Baselink)")]
    public Transform Target; // Transform to Teleport
    [Tooltip("Drone Actuator or Winch System (Rigid Body)")]
    public Transform DroneActuator;

    private Vector<double> initialPositionSAM;
    private Vector3 elevatedGoalPos;

    private Vector3 previousVelocity;

    private int immovableStage = 0;
    private float cumulativeReward = 0f;

    private float prevDistanceToGoal = 0f; 
    private Vector3 prevVelocity = Vector3.zero;

    private float currentDistanceToGoal = 0f; 
    private List<Vector3> dronePositions = new List<Vector3>();
    private string logFilePath;
    private int episodeCounter;




    public override void Initialize()
    {
        if (goal == null)
        {
            Debug.LogWarning("Target or Goal not set for DroneAgent. Disabling.");
            enabled = false;
        }

        // Check if BaseLink exists and has an ArticulationBody component
        if (BaseLink == null)
        {
            Debug.LogError("BaseLink or ArticulationBody is missing!");
            enabled = false;
            return;
        }

        initialPositionSAM = BaseLinkSAM.transform.position.To<ENU>().ToDense();

        baseLinkDroneAB = BaseLink.GetComponent<ArticulationBody>();
        ABparts = Target.gameObject.GetComponentsInChildren<ArticulationBody>();
        RBparts = DroneActuator.gameObject.GetComponentsInChildren<Rigidbody>();

        elevatedGoalPos = new Vector3(
            goal.position.x,
            goal.position.y,
            goal.position.z
        );

        string folder = "/home/shs/colcon_ws/src/smarc2/simulation/SMARCUnityStandard/Assets/Trajectory_Data";
        Directory.CreateDirectory(folder);  // Creates folder if it doesn't exist

        logFilePath = Path.Combine(folder, "drone_positions_runid3_test1.csv");

        // logFilePath = Path.Combine(Application.dataPath, "drone_positions_runid2_test2.csv");

        episodeCounter = 0; 

    }
    

    public override void CollectObservations(VectorSensor sensor)
    {   
        var position = BaseLink.transform.position;
        
        Vector3 toGoal = elevatedGoalPos - position;
        sensor.AddObservation(toGoal.normalized);    // Direction to goal (3 floats)
        sensor.AddObservation(toGoal.magnitude);    // Distance to goal (1 float)

        var velocity = baseLinkDroneAB.linearVelocity;
        sensor.AddObservation(velocity.x);
        sensor.AddObservation(velocity.y);
        sensor.AddObservation(velocity.z);

        // Debug.Log($"[DroneAgent] Observations Collected | Drone Pos : {position} | Vel: {velocity} | Acc: {acceleration}");
        // Debug.Log($"[DroneAgent] Observations Collected | Goal: {goal.position}");
        // float distanceToGoal = Vector3.Distance(BaseLink.transform.position, elevatedGoalPos);
        // Debug.Log($"[DroneAgent] Observations Collected | DISTANCE TO GOAL: {distanceToGoal}");
        
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float velX = actionBuffers.ContinuousActions[0];
        float velY = actionBuffers.ContinuousActions[1];
        float velZ = actionBuffers.ContinuousActions[2];

        // Debug.Log($"[DroneAgent] ACTION RECEIVED | Vel: ({velX}, {velY}, {velZ})");

        var position = BaseLink.transform.position;


        if (droneController != null)
        {
            droneController.TargetVelocity = new Vector3(velX, velY, velZ) * maxSpeed;
            // droneController.TargetAccel = new Vector3(accelX, accelY, accelZ)*0;
        }


        float distanceToGoal = Vector3.Distance(BaseLink.transform.position, elevatedGoalPos);
        currentDistanceToGoal = distanceToGoal;
        float distancedCoveredTowardsGoal = prevDistanceToGoal - currentDistanceToGoal;
        prevDistanceToGoal = currentDistanceToGoal;

        Vector3 currentVelocity = new Vector3(velX, velY, velZ);
        float jitterPenalty = Vector3.Distance(currentVelocity, prevVelocity);

        // Optional: scale the penalty to tune its effect
        float jitterPenaltyWeight = 0.05f;
        float jitterReward = -jitterPenalty * jitterPenaltyWeight;
        // Save for next step
        prevVelocity = currentVelocity; 

        // float maxDistance = 10f; // Maximum distance from the Drone to the goal 
        // float normalizedDistance = (1f - distanceToGoal / maxDistance); //
        // float velocityPenalty = -Vector3.Magnitude(baseLinkDroneAB.linearVelocity) * 0.01f;
        // float waterAvoidanceReward = (position.y > 0f && position.y < 2f) ? 0f : 0.5f;
        // float reward = normalizedDistance ;// + waterAvoidanceReward; // + velocityPenalty;

        float distanceFactor = 4 + 3 * math.tanh((currentDistanceToGoal - 10) / 3);
        float timePenalty = -20f / 2000; // MaxStep = 1000 typically
        float reward = distancedCoveredTowardsGoal * 50 * distanceFactor + timePenalty + jitterReward;

        AddReward(reward);
        cumulativeReward += reward;

        if (debugMode) Debug.Log($"[DroneAgent] Distance to Goal: {distanceToGoal} | Reward: {reward}");
        if (debugMode) Debug.Log($"Distance reward: {distancedCoveredTowardsGoal * 50 * distanceFactor} | Time Penalty: {timePenalty} | Jittery Movement Reward: {jitterReward}");
        if (debugMode) Debug.Log($"Total Episode: {episodeCounter}");
    }

    public void FixedUpdate()
    {
        // Debug.Log("[DroneAgent] Method : E N T E R E D    I N T O    F I X E D U P D A T E");


        float distanceToGoal = Vector3.Distance(BaseLink.transform.position, elevatedGoalPos);
        // Debug.Log($"[DroneAgent] Distance to Goal: {distanceToGoal} ");
        var position = BaseLink.transform.position;

        dronePositions.Add(BaseLink.transform.position);

        if (distanceToGoal < 0.3f)
        {
            AddReward(10f);
            WritePositionsToFile();
            EndEpisode();
            Debug.Log($"EPISODE ENDED, *******GOAL REACHED*******, Distance to Goal:  {distanceToGoal}");
        }

        else if (distanceToGoal > 15f)
        {
            AddReward(-100f);
            WritePositionsToFile();
            EndEpisode();
            Debug.Log($"EPISODE ENDED, DRONE IS TOO FAR, Distance to Goal: {distanceToGoal}");
        }

        else if (position.y < 0f)
        {
            SetReward(-100f);
            WritePositionsToFile();
            EndEpisode();
            Debug.Log($"EPISODE ENDED, DRONE WENT UNDERWATER, Distance to Goal: {distanceToGoal}");
        }

    }

    public override void OnEpisodeBegin()
    {
        if (debugMode) Debug.Log("[DroneAgent] On Episode Beginning. Resetting agent...");

        dronePositions.Clear();
        episodeCounter++;


        immovableStage = 0;

        for (int counter = 0; counter < 3; counter++)
        {
            switch (immovableStage)
            {
                case 0:
                    ResetPosition();
                    immovableStage = 1;
                    break;
                case 1:
                    if (Target.TryGetComponent(out ArticulationBody targetAb))
                    {
                        if (!targetAb.isRoot) return;
                        // targetAb.immovable = false;
                        // Debug.Log("IMMOVABLE STAGE WORKED");
                    }
                    immovableStage = 2;
                    break;
                default:
                    break;
            }
        }

        if (debugMode) Debug.Log("******************* A G E N T   R E S E T ************************");
        if (debugMode) Debug.Log($" [ResetAgent] Goal Position  : {elevatedGoalPos.x}, {elevatedGoalPos.y}, {elevatedGoalPos.z}");
        if (debugMode) Debug.Log($" [ResetAgent] Drone Position : {BaseLink.transform.position.x}, {BaseLink.transform.position.y}, {BaseLink.transform.position.z}");
        
        

    }

    void WritePositionsToFile()
    {
        using (StreamWriter writer = new StreamWriter(logFilePath, true)) // true to append
        {
            writer.WriteLine("New Episode:");
            foreach (Vector3 pos in dronePositions)
            {
                if (episodeCounter % 5 == 0) writer.WriteLine($"{episodeCounter},{pos.x},{pos.y},{pos.z}");
            }
            writer.WriteLine(); // blank line between episodes
        }
    }



    void ResetPosition()
    {   
        
        // Use the initial position from BaseLink and convert it properly
        var NewPosition = ENU.ConvertToRUF(
            new Vector3(
                (float)initialPositionSAM[0]+5f + Random.Range(-5f,5f),  
                (float)initialPositionSAM[1]+8f,
                (float)initialPositionSAM[2]+7f + Random.Range(-2f,2f) //keeping the position same as the initial position of the drone 
            ));
         // Use a default orientation (identity quaternion)
        var NewOrientation = Quaternion.identity;


        if (Target.TryGetComponent(out ArticulationBody targetAb))
        {
            if (!targetAb.isRoot) return;
            // targetAb.immovable = true;
            immovableStage = 0;
            targetAb.TeleportRoot(NewPosition, NewOrientation);
            targetAb.linearVelocity = Vector3.zero;
            targetAb.angularVelocity = Vector3.zero;
            targetAb.linearDamping = 0f;
        }
        else
        {
            Debug.Log("Target is not an Articulation Body");
        }


        foreach (var ab in ABparts)
        {
            ab.linearVelocity = Vector3.zero;
            ab.angularVelocity = Vector3.zero;
            ab.ResetArticulationBody();
        }


        for (int i = 0; i < RBparts.Length; i++)
        {   
            // Reset position and rotation to initial values
            RBparts[i].transform.position = NewPosition; // initialPositionWinch[i];
            RBparts[i].rotation = Quaternion.Euler(0f, 0f, 0f);

            // Reset velocity to stop movement
            RBparts[i].linearVelocity = Vector3.zero;
            RBparts[i].angularVelocity = Vector3.zero;
        }  

        if(debugMode) Debug.Log("******************* A G E N T   I S    R E S E T ************************");          

    }
}
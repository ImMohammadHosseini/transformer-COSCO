from simulator.host.Host import *
from simulator.container.Container import *
from copy import deepcopy

class Simulator():
	# Total power in watt
	# Total Router Bw
	# Interval Time in seconds
    def __init__(self, TotalPower, RouterBw, Scheduler, ContainerLimit, IntervalTime, hostinit):
        self.totalpower = TotalPower
        self.totalbw = RouterBw
        self.hostlimit = len(hostinit)
        self.scheduler = Scheduler
        self.scheduler.setEnvironment(self)
        self.containerlimit = ContainerLimit
        self.hostlist = []
        self.containerlist = []
        self.intervaltime = IntervalTime
        self.interval = 0
        self.inactiveContainers = []
        self.stats = None
        self.addHostlistInit(hostinit)

    def reset (self, hostinit):
        self.hostlimit = len(hostinit)
        self.scheduler.setEnvironment(self)
        self.hostlist = []
        self.containerlist = []
        self.interval = 0
        self.inactiveContainers = []
        self.stats = None
        self.addHostlistInit(hostinit)
        
    def addHostInit(self, IPS, RAM, Disk, Bw, Latency, Powermodel):
        assert len(self.hostlist) < self.hostlimit
        host = Host(len(self.hostlist), IPS, RAM, Disk, Bw, Latency, Powermodel, self)
        self.hostlist.append(host)

    def addHostlistInit(self, hostList):
        assert len(hostList) == self.hostlimit
        for IPS, RAM, Disk, Bw, Latency, Powermodel in hostList:
            self.addHostInit(IPS, RAM, Disk, Bw, Latency, Powermodel)

    def addContainerInit(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
        container = Container(len(self.containerlist), CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID = -1)
        self.containerlist.append(container)
        return container

    def addContainerListInit(self, containerInfoList):
        deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())]
        deployedContainers = []
        for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
            dep = self.addContainerInit(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
            deployedContainers.append(dep)
        self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
        return [container.id for container in deployedContainers]

    def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
        for i,c in enumerate(self.containerlist):
            if c == None or not c.active:
                container = Container(i, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID = -1)
                self.containerlist[i] = container
                return container

    def addContainerList(self, containerInfoList):
        deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit-self.getNumActiveContainers())]
        deployedContainers = []
        for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
            dep = self.addContainer(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
            deployedContainers.append(dep)
        return [container.id for container in deployedContainers]

    def getContainersOfHost(self, hostID):
        containers = []
        for container in self.containerlist:
            if container and container.hostid == hostID:
                containers.append(container.id)
        return containers

    def getContainerByID(self, containerID):
        return self.containerlist[containerID]

    def getContainerByCID(self, creationID):
        for c in self.containerlist + self.inactiveContainers:
            if c and c.creationID == creationID:
                return c

    def getHostByID(self, hostID):
        return self.hostlist[hostID]

    def getCreationIDs(self, migrations, containerIDs):
        creationIDs = []
        for decision in migrations:
            if decision[0] in containerIDs: creationIDs.append(self.containerlist[decision[0]].creationID)
        return creationIDs

    def getPlacementPossible(self, containerID, hostID):
        container = self.containerlist[containerID]
        host = self.hostlist[hostID]
        ipsreq = container.getBaseIPS()
        ramsizereq, ramreadreq, ramwritereq = container.getRAM()
        disksizereq, diskreadreq, diskwritereq = container.getDisk()
        ipsavailable = host.getIPSAvailable()
        ramsizeav, ramreadav, ramwriteav = host.getRAMAvailable()
        disksizeav, diskreadav, diskwriteav = host.getDiskAvailable()
        return (ipsreq <= ipsavailable and \
				ramsizereq <= ramsizeav and \
				# ramreadreq <= ramreadav and \
				# ramwritereq <= ramwriteav and \
				disksizereq <= disksizeav \
				# diskreadreq <= diskreadav and \
				# diskwritereq <= diskwriteav
				)

    def addContainersInit(self, containerInfoListInit):
        self.interval += 1
        deployed = self.addContainerListInit(containerInfoListInit)
        return deployed

    def allocateInit(self, decision):
        migrations = []
        rewards = {}
        routerBwToEach = self.totalbw / (len(decision) + 1e-10)
        for (cid, hid) in decision:
            container = self.getContainerByID(cid)
            assert container.getHostID() == -1
            numberAllocToHost = len(self.scheduler.getMigrationToHost(hid, decision))
            allocbw = min(self.getHostByID(hid).bwCap.downlink / numberAllocToHost, routerBwToEach)
            if self.getPlacementPossible(cid, hid):
                if container.getHostID() != hid:
                    migrations.append((cid, hid))
                container.allocateAndExecute(hid, allocbw)
                
                predictExecTime = container.totalExecTime+container.totalMigrationTime+(container.execTimeAfterMigration*container.ipsmodel.getTotalInstructions())/(container.ipsmodel.completedAfterMigration)
                rewards[cid, hid] = [1e4*(1/predictExecTime)*((container.createAt+1)/(self.interval+1))]
                #if container.getBaseIPS() == 0:
                #    ten_scale = 10*(container.ipsmodel.completedAfterMigration / container.ipsmodel.getTotalInstructions())
                #    rewards[container.id, container.hostid] = ((container.createAt+1)/(self.interval+1))*ten_scale 

			# destroy pointer to this unallocated container as book-keeping is done by workload model
            else: 
                self.containerlist[cid] = None
        return migrations, rewards

    def destroyCompletedContainers(self):
        destroyed = []
        for i,container in enumerate(self.containerlist):
            if container and container.getBaseIPS() == 0:
                container.destroy()
                self.containerlist[i] = None
                self.inactiveContainers.append(container)
                destroyed.append(container)
        return destroyed

    def getNumActiveContainers(self):
        num = 0 
        for container in self.containerlist:
            if container and container.active: num += 1
        return num

    def getSelectableContainers(self):
        selectable = []
        for container in self.containerlist:
            if container and container.active and container.getHostID() != -1:
                selectable.append(container.id)
        return selectable

    def addContainers(self, newContainerList):
        self.interval += 1
        destroyed = self.destroyCompletedContainers()
        deployed = self.addContainerList(newContainerList)
        return deployed, destroyed

    def getActiveContainerList(self):
        return [c.getHostID() if c and c.active else -1 for c in self.containerlist]

    def getContainersInHosts(self):
        return [len(self.getContainersOfHost(host)) for host in range(self.hostlimit)]

    def simulationStep(self, decision):
        routerBwToEach = self.totalbw / len(decision) if len(decision) > 0 else self.totalbw
        migrations = []
        containerIDsAllocated = []
        rewards = {}
        for (cid, hid) in decision:
            container = self.getContainerByID(cid)
            currentHostID = self.getContainerByID(cid).getHostID()
            currentHost = self.getHostByID(currentHostID)
            targetHost = self.getHostByID(hid)
            migrateFromNum = len(self.scheduler.getMigrationFromHost(currentHostID, decision))
            migrateToNum = len(self.scheduler.getMigrationToHost(hid, decision))
            allocbw = min(targetHost.bwCap.downlink / migrateToNum, currentHost.bwCap.uplink / migrateFromNum, routerBwToEach)
            placementCondition = self.getPlacementPossible(cid, hid)
            if hid != self.containerlist[cid].hostid and placementCondition:
                #if container.hostid != -1:
                #    ten_scale = 10*(container.ipsmodel.completedAfterMigration / container.ipsmodel.getTotalInstructions())
                #    rewards[cid, container.hostid] = ((container.createAt+1)/(self.interval+1))*ten_scale
                containerRemainInstruction = container.ipsmodel.getTotalInstructions()-container.ipsmodel.completedInstructions
                firstAllocation = True
                if container.hostid != -1:
                    firstAllocation = False
                    oldExecTime, oldCompletedInstructions = container.semi_execute()
                    predictOldExecTime = container.totalExecTime+container.totalMigrationTime+((oldExecTime*containerRemainInstruction)/oldCompletedInstructions if oldCompletedInstructions else 0)
                    predictOldExecTime = predictOldExecTime if predictOldExecTime > 1e-1 else 0
                migrations.append((cid, hid))
                container.allocateAndExecute(hid, allocbw)
                
                completedAfterMigration = container.ipsmodel.completedAfterMigration
                predictExecTime = container.totalExecTime+container.totalMigrationTime+((container.execTimeAfterMigration*containerRemainInstruction)/completedAfterMigration if completedAfterMigration else 0)
                predictExecTime = predictExecTime if predictExecTime > 1e-1 else 0
                if predictExecTime == 0: rewards[cid, hid] = [0]
                elif not firstAllocation and predictOldExecTime != 0:
                    #print('reward1', 1000*((1/predictExecTime) - (1/predictOldExecTime)))
                    rewards[cid, hid] = [1e4*((1/predictExecTime) - (1/predictOldExecTime))*((container.createAt+1)/(self.interval+1))]
                    #print('22', rewards[cid, hid])
                else: 
                    #print('reward2', 1000*(1/predictExecTime))
                    rewards[cid, hid] = [1e4*(1/predictExecTime)*((container.createAt+1)/(self.interval+1))]
                
                #if container.getBaseIPS() == 0:
                #    ten_scale = 10*(container.ipsmodel.completedAfterMigration / container.ipsmodel.getTotalInstructions())
                #    rewards[container.id, container.hostid] = ((container.createAt+1)/(self.interval+1))*ten_scale 
                containerIDsAllocated.append(cid)
                
            elif not placementCondition:
                rewards[cid, hid] = [0]
                
		# destroy pointer to unallocated containers as book-keeping is done by workload model
        '''for (cid, hid) in decision:
            if self.containerlist[cid].hostid == -1: 
                self.containerlist[cid] = None'''
        
        for i,container in enumerate(self.containerlist):
           
            if container and self.containerlist[i].hostid == -1:
                #print('none in exe', i)
                #print('getTotalInstructions',container.ipsmodel.getTotalInstructions())
                #print(container.ipsmodel.completedInstructions)
                self.containerlist[i] = None
            
        for i,container in enumerate(self.containerlist):
            if container and i not in containerIDsAllocated:
                container.execute(0)
                #if container.getBaseIPS() == 0:
                #    ten_scale = 10*(container.ipsmodel.completedAfterMigration / container.ipsmodel.getTotalInstructions())
                #    rewards[container.id, container.hostid] = ((container.createAt+1)/(self.interval+1))*ten_scale 

        return migrations, rewards
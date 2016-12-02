class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        buff={}
        for i in range(len(nums)):
            if nums[i] in buff:
                return[buff[nums[i]],i+1]
            else :
                buff[target-nums[i]]=i+1

local TransposeX, parent = torch.class('nn.TransposeX', 'nn.Module')

function TransposeX:__init()
   parent.__init(self)
end

function TransposeX:updateOutput(input)
  self.output = {}
  for i, k in ipairs(input) do
    local channel = k:size()[1]
    local height  = k:size()[2]
    local width   = k:size()[3]
    assert( height == 1, 'height must be 1')
    table.insert(self.output, k:resize(channel,width):t())
  end
  return self.output
end

function TransposeX:updateGradInput(input, gradOutput)
   gradOutput = threeDtable2tensor(gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   return self.gradInput
end

function threeDtable2tensor(tab)
		local batch = #tab
    local length = tab[1]:size()[1]
    local feature = tab[1]:size()[2]
		local tensor = torch.Tensor(batch,feature,length):fill(0)
    for key,val in ipairs(tab) do
      tensor[key] = val:t()
    end
		return tensor:resize(batch,1,feature,length)
end
